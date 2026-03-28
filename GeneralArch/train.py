"""Training script for the masking-agnostic ResNet SCA model.

Follows the methodology from Marquet & Oswald (CHES 2024):
  - Multi-task: all bytes simultaneously
  - Low-level parameter sharing (m_d design)
  - Multi-target (s1 + t1) option
  - 3-phase LR schedule as in the reference code

Usage:
    python train.py --dataset ascad_r --model_name baseline_v1 --epochs 100
    python train.py --dataset ascad_v2 --model_name baseline_v2 --epochs 100 --multi_target
"""

import argparse
import os
import pickle
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from dataset import build_tf_dataset, DATASET_CONFIGS
from model import build_model


def get_output_dirs(dataset_name):
    """Return (model_dir, metrics_dir)."""
    cfg = DATASET_CONFIGS[dataset_name]
    base = cfg["folder"]
    model_dir = base + "models/"
    metrics_dir = base + "metrics/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    return model_dir, metrics_dir


class ThreePhaseSchedule(tf.keras.callbacks.Callback):
    """3-phase LR schedule matching the reference code:
       Phase 1: lr      for `phase1` epochs
       Phase 2: lr/10   for `phase2` epochs
       Phase 3: lr/100  for remaining epochs
    """

    def __init__(self, lr, phase1=10, phase2=25):
        super().__init__()
        self.lr = lr
        self.phase1 = phase1
        self.phase2 = phase2

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.phase1:
            new_lr = self.lr
        elif epoch < self.phase1 + self.phase2:
            new_lr = self.lr / 10
        else:
            new_lr = self.lr / 100
        self.model.optimizer.learning_rate.assign(new_lr)
        if epoch in (0, self.phase1, self.phase1 + self.phase2):
            print(f"\n  LR set to {new_lr:.6f} at epoch {epoch}")


class EpochSummary(tf.keras.callbacks.Callback):
    """Print one compact line per epoch: mean acc, mean val acc, loss, val loss."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Gather per-output accuracies and losses
        train_accs = [v for k, v in logs.items()
                      if k.endswith("_accuracy") and not k.startswith("val_")]
        val_accs = [v for k, v in logs.items()
                    if k.startswith("val_") and k.endswith("_accuracy")]
        train_loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        mean_acc = np.mean(train_accs) if train_accs else float("nan")
        mean_val_acc = np.mean(val_accs) if val_accs else float("nan")
        lr = float(self.model.optimizer.learning_rate)
        print(f"Epoch {epoch + 1:3d} | "
              f"loss: {train_loss:.4f}  acc: {mean_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {mean_val_acc:.4f} | "
              f"lr: {lr:.6f}")


def train(args):
    # Seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    cfg = DATASET_CONFIGS[args.dataset]
    byte_range = cfg["byte_range"]
    model_dir, metrics_dir = get_output_dirs(args.dataset)

    print(f"=== Training on {args.dataset} | seed={args.seed} ===")
    print(f"Bytes: {list(byte_range)} | Multi-target: {args.multi_target}")
    lr_mode = "static" if args.static_lr else f"3-phase ({args.phase1}/{args.phase2})"
    es_mode = f"patience={args.patience}" if args.patience > 0 else "off"
    print(f"Traces: {args.n_traces or 'all'} | Epochs: {args.epochs}")
    print(f"LR: {args.learning_rate} ({lr_mode}) | Early stopping: {es_mode}")
    reg_parts = []
    if args.l2_reg > 0:
        reg_parts.append(f"L2={args.l2_reg}")
    if args.dropout_rate > 0:
        reg_parts.append(f"dropout={args.dropout_rate}")
    if args.noise_std > 0:
        reg_parts.append(f"noise={args.noise_std}")
    if args.use_attention:
        reg_parts.append(f"attention={args.attention_heads}heads")
    if reg_parts:
        print(f"Regularization: {', '.join(reg_parts)}")

    # Load data
    train_ds, val_ds, trace_length = build_tf_dataset(
        args.dataset,
        n_traces=args.n_traces,
        batch_size=args.batch_size,
        byte_range=byte_range,
        multi_target=args.multi_target,
        noise_std=args.noise_std,
    )

    # Build model
    if args.n_components > 1:
        print(f"Combiner mode: {args.n_components} components, "
              f"bilinear rank={args.combiner_rank}")
    model = build_model(
        input_length=trace_length,
        byte_range=byte_range,
        convolution_blocks=args.convolution_blocks,
        filters=args.filters,
        kernel_size=args.kernel_size,
        strides=args.strides,
        pooling_size=args.pooling_size,
        dense_units=args.dense_units,
        non_shared_blocks=args.non_shared_blocks,
        shared_blocks=args.shared_blocks,
        n_components=args.n_components,
        combiner_rank=args.combiner_rank,
        combiner_skip=args.combiner_skip,
        l2_reg=args.l2_reg,
        dropout_rate=args.dropout_rate,
        use_attention=args.use_attention,
        attention_heads=args.attention_heads,
        multi_target=args.multi_target,
        learning_rate=args.learning_rate,
        seed=args.seed,
        summary=False,
    )

    # Callbacks
    callbacks = []

    # LR schedule (optional)
    if not args.static_lr:
        callbacks.append(ThreePhaseSchedule(
            args.learning_rate, phase1=args.phase1, phase2=args.phase2,
        ))

    # Save best weights on val loss
    ckpt_path = os.path.join(model_dir, f"{args.model_name}_best.weights.h5")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True,
        save_weights_only=True, verbose=1,
    ))

    # Early stopping
    if args.patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ))

    # Compact epoch logging
    callbacks.append(EpochSummary())

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )

    # Save final weights + all weights
    final_path = os.path.join(model_dir, f"{args.model_name}_final.weights.h5")
    model.save_weights(final_path)

    all_path = os.path.join(model_dir, f"{args.model_name}_all.weights.h5")
    model.save_weights(all_path)

    print(f"Best  weights: {ckpt_path}")
    print(f"Final weights: {final_path}")

    # Save history
    hist_path = os.path.join(metrics_dir, f"{args.model_name}_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"History: {hist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train masking-agnostic ResNet for SCA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(__import__('dataset').DATASET_CONFIGS.keys()))
    parser.add_argument("--n_traces", type=int, default=None,
                        help="Training traces (None=all)")
    parser.add_argument("--batch_size", type=int, default=250)

    # ResNet backbone (matching reference defaults)
    parser.add_argument("--convolution_blocks", type=int, default=1)
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=34)
    parser.add_argument("--strides", type=int, default=17)
    parser.add_argument("--pooling_size", type=int, default=2)

    # Prediction heads
    parser.add_argument("--dense_units", type=int, default=200)
    parser.add_argument("--non_shared_blocks", type=int, default=1)
    parser.add_argument("--shared_blocks", type=int, default=1)
    parser.add_argument("--multi_target", action="store_true", default=False,
                        help="Predict both s1 and t1")

    # Learnable combiner (masking-agnostic decomposition)
    parser.add_argument("--n_components", type=int, default=1,
                        help="Component branches (1=direct, 2=bilinear combiner)")
    parser.add_argument("--combiner_rank", type=int, default=64,
                        help="Rank of bilinear CP decomposition (higher=more expressive)")
    parser.add_argument("--combiner_skip", action="store_true", default=False,
                        help="Add residual skip connection around combiner")

    # Regularization
    parser.add_argument("--l2_reg", type=float, default=0.0,
                        help="L2 regularization strength (0=off)")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="SpatialDropout1D rate after ResNet blocks (0=off)")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Gaussian noise std on normalized traces (0=off)")

    # Self-attention
    parser.add_argument("--use_attention", action="store_true", default=False,
                        help="Add self-attention after ResNet backbone")
    parser.add_argument("--attention_heads", type=int, default=4,
                        help="Number of attention heads")

    # Training
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--phase1", type=int, default=10,
                        help="Epochs at full LR")
    parser.add_argument("--phase2", type=int, default=25,
                        help="Epochs at LR/10")
    parser.add_argument("--static_lr", action="store_true", default=False,
                        help="Use constant LR (no 3-phase schedule)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0=off)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
