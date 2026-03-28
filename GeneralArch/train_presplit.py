"""
Pre-split ASCAD_v2 combiner test.

This script tests whether the BilinearCombinerLayer can learn the affine
demasking operation when the backbone's job is eliminated by pre-splitting
the trace into known regions (matching the original paper's architecture).

If this works → the combiner handles affine, the backbone is the bottleneck.
If this fails → the combiner itself needs more capacity (3 components, higher rank).

Trace regions (from the original paper):
    alpha:  traces[:, :2000]      — 2000 points, leaks alpha (GF(256) multiplier)
    rin:    traces[:, 2000:3000]  — 1000 points, leaks intermediate mask
    beta:   traces[:, 3000:3200]  —  200 points, leaks beta (additive mask)
    block:  traces[:, 3200:4688]  — 1488 points (93x16), leaks masked AES state

Each region gets its own small Conv1D backbone (matching the paper's specs).
The 4 backbone outputs are concatenated and fed into per-byte heads with
BilinearCombinerLayer + skip — the same combiner that works on ASCAD_r.
"""

import argparse
import os
import sys
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dense, AveragePooling1D,
    Flatten, Softmax, Concatenate, LayerNormalization, Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers, regularizers

# Import custom layers from the existing GeneralArch code
sys.path.insert(0, os.path.dirname(__file__))
from model import BilinearCombinerLayer, SharedWeightsDenseLayer

AES_SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)


# ── data loading ──────────────────────────────────────────────────────

H5_PATH = "/home/projects/sipl-prj10622/ASCAD/ASCAD_v2/Ascad_v2_extracted_exp_experiments.h5"

REGION_SLICES = {
    "alpha": (0, 2000),
    "rin":   (2000, 3000),
    "beta":  (3000, 3200),
    "block": (3200, 4688),
}


def _normalize(traces):
    mean = np.mean(traces, axis=1, keepdims=True)
    std = np.std(traces, axis=1, keepdims=True) + 1e-8
    return ((traces - mean) / std).astype(np.float32)


def load_split(split, n_traces=None):
    f = h5py.File(H5_PATH, "r")
    grp = f[split]
    n = n_traces if n_traces else grp["traces"].shape[0]
    raw = np.array(grp["traces"][:n])
    s1 = np.array(grp["labels"]["s1"][:n], dtype=np.uint8)
    plaintexts = np.array(grp["plaintexts"][:n], dtype=np.uint8)
    keys = np.array(grp["keys"][:n], dtype=np.uint8)
    f.close()

    # Split and normalize each region independently
    regions = {}
    for name, (start, end) in REGION_SLICES.items():
        r = _normalize(raw[:, start:end].astype(np.float32))
        regions[name] = r[..., np.newaxis]  # add channel dim

    return regions, s1, plaintexts, keys


def build_datasets(n_traces, batch_size, noise_std):
    regions_train, s1_train, _, _ = load_split("training", n_traces)
    regions_val, s1_val, _, _ = load_split("validation", 10000)

    print(f"Training: {s1_train.shape[0]} traces")
    print(f"Validation: {s1_val.shape[0]} traces")
    for name, arr in regions_train.items():
        print(f"  {name}: {arr.shape[1]} points")

    eye256 = np.eye(256, dtype=np.float32)

    x_train = {f"input_{k}": v for k, v in regions_train.items()}
    x_val = {f"input_{k}": v for k, v in regions_val.items()}
    y_train, y_val = {}, {}
    for b in range(16):
        y_train[f"output_{b}"] = eye256[s1_train[:, b]]
        y_val[f"output_{b}"] = eye256[s1_val[:, b]]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(s1_train), 50000), reshuffle_each_iteration=True)
        .batch(batch_size)
    )
    if noise_std > 0:
        def _augment(x, y):
            for k in x:
                x[k] = x[k] + tf.random.normal(tf.shape(x[k]), stddev=noise_std)
            return x, y
        train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


# ── model ─────────────────────────────────────────────────────────────

def cnn_backbone(input_tensor, filters, kernel_size, strides, pool_size,
                 l2_reg, seed, name_prefix):
    """Small Conv1D backbone for a single trace region."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    x = Conv1D(
        filters, kernel_size, strides=strides, padding="same",
        activation="selu", kernel_regularizer=reg,
        kernel_initializer=initializers.RandomUniform(seed=seed),
        name=f"{name_prefix}_conv",
    )(input_tensor)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    if pool_size > 1:
        x = AveragePooling1D(pool_size=pool_size, name=f"{name_prefix}_pool")(x)
    x = Flatten(name=f"{name_prefix}_flat")(x)
    return x


def build_presplit_model(
    n_components=2,
    combiner_rank=64,
    combiner_skip=True,
    dense_units=200,
    shared_blocks=1,
    l2_reg=0.0001,
    learning_rate=0.001,
    seed=42,
):
    """Build the pre-split combiner test model.

    4 separate backbone inputs → concatenate → per-byte heads → combiner.
    Backbone specs match the original paper (Conv1D + Dense per region).
    """
    n_bytes = 16
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # ── 4 input branches ──
    input_alpha = Input(shape=(2000, 1), name="input_alpha")
    input_rin = Input(shape=(1000, 1), name="input_rin")
    input_beta = Input(shape=(200, 1), name="input_beta")
    input_block = Input(shape=(1488, 1), name="input_block")

    # ── Per-region Conv1D backbones (matching paper specs) ──
    # Paper: alpha: k=32, f=16, s=10, pool=5
    alpha_feat = cnn_backbone(input_alpha, filters=16, kernel_size=32,
                              strides=10, pool_size=5, l2_reg=l2_reg,
                              seed=seed, name_prefix="alpha")
    # Paper: rin: k=32, f=16, s=5, pool=5
    rin_feat = cnn_backbone(input_rin, filters=16, kernel_size=32,
                            strides=5, pool_size=5, l2_reg=l2_reg,
                            seed=seed, name_prefix="rin")
    # Paper: beta: k=32, f=16, s=1, pool=5
    beta_feat = cnn_backbone(input_beta, filters=16, kernel_size=32,
                             strides=1, pool_size=5, l2_reg=l2_reg,
                             seed=seed, name_prefix="beta")
    # Block: use similar backbone (k=32, f=16, s=10, pool=5)
    block_feat = cnn_backbone(input_block, filters=16, kernel_size=32,
                              strides=10, pool_size=5, l2_reg=l2_reg,
                              seed=seed, name_prefix="block")

    # ── Concatenate all region features ──
    backbone = Concatenate(name="backbone_concat")([
        alpha_feat, rin_feat, beta_feat, block_feat
    ])
    # backbone shape: (batch, total_features)

    print(f"Backbone feature dim: alpha={alpha_feat.shape[1]}, "
          f"rin={rin_feat.shape[1]}, beta={beta_feat.shape[1]}, "
          f"block={block_feat.shape[1]}, "
          f"total={backbone.shape[1]}")

    # ── Per-byte heads (same as GeneralArch) ──
    def build_component_head(backbone_in, comp_name):
        """Per-byte Dense → SharedWeightsDenseLayer → logits."""
        per_byte = []
        for b in range(n_bytes):
            h = Dense(
                dense_units, activation="selu", kernel_regularizer=reg,
                kernel_initializer=initializers.RandomUniform(seed=seed),
                name=f"{comp_name}_byte{b}_dense",
            )(backbone_in)
            per_byte.append(Reshape((dense_units, 1))(h))

        stacked = Concatenate(axis=2, name=f"{comp_name}_stack")(per_byte)
        # (batch, dense_units, 16)

        x = stacked
        for blk in range(shared_blocks):
            x = SharedWeightsDenseLayer(
                input_dim=x.shape[1], units=dense_units, shares=n_bytes,
                activation=True, seed=seed,
                name=f"{comp_name}_shared_{blk}",
            )(x)

        logits = SharedWeightsDenseLayer(
            input_dim=dense_units, units=256, shares=n_bytes,
            activation=False, seed=seed,
            name=f"{comp_name}_logits",
        )(x)
        return logits  # (batch, 256, 16)

    inputs = {
        "input_alpha": input_alpha,
        "input_rin": input_rin,
        "input_beta": input_beta,
        "input_block": input_block,
    }

    if n_components == 1:
        # Direct prediction — no combiner
        logits = build_component_head(backbone, "direct")
    elif n_components == 2:
        logits_a = build_component_head(backbone, "comp_a")
        logits_b = build_component_head(backbone, "comp_b")

        logits_a_norm = LayerNormalization(axis=1, name="ln_a")(logits_a)
        logits_b_norm = LayerNormalization(axis=1, name="ln_b")(logits_b)

        combined = BilinearCombinerLayer(
            rank=combiner_rank, n_classes=256, shares=n_bytes,
            seed=seed, name="bilinear_combiner",
        )([logits_a_norm, logits_b_norm])

        if combiner_skip:
            logits = combined + logits_a
        else:
            logits = combined
    else:
        raise ValueError(f"n_components={n_components} not supported (use 1 or 2)")

    # ── Per-byte softmax outputs ──
    outputs = {}
    metrics = {}
    losses = {}
    for b in range(n_bytes):
        out = Softmax(name=f"output_{b}")(logits[:, :, b])
        outputs[f"output_{b}"] = out
        metrics[f"output_{b}"] = "accuracy"
        losses[f"output_{b}"] = "categorical_crossentropy"

    model = Model(inputs=inputs, outputs=outputs, name="presplit_combiner")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=losses,
        metrics=metrics,
    )
    return model


# ── training ──────────────────────────────────────────────────────────

class EpochSummary(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        # Average accuracy across all byte outputs
        accs = [v for k, v in logs.items() if k.startswith("output_") and k.endswith("accuracy") and "val" not in k]
        val_accs = [v for k, v in logs.items() if k.startswith("val_output_") and k.endswith("accuracy")]
        acc = np.mean(accs) if accs else 0
        val_acc = np.mean(val_accs) if val_accs else 0
        lr = float(self.model.optimizer.learning_rate)
        print(f"Epoch {epoch+1:3d} | loss: {loss:.4f}  acc: {acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | lr: {lr:.6f}")


def train(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== Pre-split combiner test | seed={args.seed} ===")
    print(f"Components: {args.n_components}, Rank: {args.combiner_rank}, "
          f"Skip: {args.combiner_skip}")
    print(f"Traces: {args.n_traces or 'all'} | Epochs: {args.epochs}")
    print(f"LR: {args.learning_rate} ({'static' if args.static_lr else 'ReduceLR'})")
    print(f"Regularization: L2={args.l2_reg}, noise={args.noise_std}")
    print()

    train_ds, val_ds = build_datasets(
        args.n_traces, args.batch_size, args.noise_std,
    )

    model = build_presplit_model(
        n_components=args.n_components,
        combiner_rank=args.combiner_rank,
        combiner_skip=args.combiner_skip,
        dense_units=args.dense_units,
        shared_blocks=args.shared_blocks,
        l2_reg=args.l2_reg,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print(f"Model parameters: {model.count_params():,}")
    print()

    weights_dir = os.path.dirname(H5_PATH) + "/models/"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = weights_dir + f"{args.model_name}_best.weights.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            weights_path, monitor="val_loss",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        EpochSummary(),
    ]

    if not args.static_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=20,
                min_lr=1e-6, verbose=1,
            )
        )

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, verbose=0, callbacks=callbacks,
    )

    print(f"Best weights: {weights_path}")
    print("$ Done !")

    # ── Attack ──
    print("\n=== Attack phase ===")
    model.load_weights(weights_path)

    regions_atk, s1_atk, pt_atk, keys_atk = load_split("attack", 5000)
    x_atk = {f"input_{k}": v for k, v in regions_atk.items()}
    preds_raw = model.predict(x_atk, verbose=0)

    # Stack predictions: (n_traces, 16, 256)
    n_traces_atk = s1_atk.shape[0]
    preds = np.zeros((n_traces_atk, 16, 256), dtype=np.float64)
    for b in range(16):
        preds[:, b, :] = preds_raw[f"output_{b}"]

    # Per-byte accuracy
    print("\n=== Per-byte s1 accuracy ===")
    for b in range(16):
        acc = np.mean(np.argmax(preds[:, b, :], axis=1) == s1_atk[:, b])
        print(f"  Byte {b:2d}: {acc*100:.2f}%")

    # Log-prob attack (variable key)
    print("\n=== Variable-key attack (1000 experiments) ===")
    n_exp = min(1000, n_traces_atk - 100)
    for exp_idx in range(0, n_exp, 100):
        t_start = exp_idx
        n_atk = 100
        scores = np.zeros((16, 256), dtype=np.float64)
        for t in range(t_start, t_start + n_atk):
            key_t = keys_atk[t]
            for b in range(16):
                for k_guess in range(256):
                    s1_guess = AES_SBOX[pt_atk[t, b] ^ k_guess]
                    scores[b, k_guess] += np.log(preds[t, b, s1_guess] + 1e-36)
            # Check GE at this point
            ge_bits = 0
            for b in range(16):
                sorted_k = np.argsort(scores[b])[::-1]
                rank = int(np.where(sorted_k == key_t[b])[0][0]) + 1
                ge_bits += np.log2(rank) if rank > 1 else 0
            if t == t_start:
                first_ge = ge_bits
        print(f"  Exp {exp_idx}: first_trace GE_sum=2^{first_ge:.0f}, "
              f"after {n_atk} traces GE_sum=2^{ge_bits:.0f}")

    print("\nAttack completed.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--combiner_rank", type=int, default=64)
    p.add_argument("--combiner_skip", action="store_true", default=False)
    p.add_argument("--dense_units", type=int, default=200)
    p.add_argument("--shared_blocks", type=int, default=1)
    p.add_argument("--l2_reg", type=float, default=0.0001)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--static_lr", action="store_true", default=False)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--n_traces", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
