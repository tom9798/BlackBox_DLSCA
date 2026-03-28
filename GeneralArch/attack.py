"""Key-recovery attack using the trained masking-agnostic model.

Supports:
  - Fixed key (ASCAD_r): accumulate log-probs across traces → GE
  - Variable key (ASCAD_v2): per-trace key ranking → mean rank

Attack: For each key candidate k, s1_hyp = Sbox(plaintext XOR k).
No XOR/GF256 layers in the model — only standard AES key-addition here.

Usage:
    python attack.py --dataset ascad_r  --model_name baseline_v1
    python attack.py --dataset ascad_v2 --model_name baseline_v2
"""

import argparse
import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from dataset import load_traces_and_labels, DATASET_CONFIGS, _normalize_traces
from model import build_model
from aes_utils import SBOX


def _pow_rank(x):
    if x <= 1:
        return 0
    n = 0
    v = int(x)
    while v > 1:
        v >>= 1
        n += 1
    return n


def _rank_of(scores, correct):
    sorted_keys = np.argsort(scores)[::-1]
    return int(np.where(sorted_keys == correct)[0][0]) + 1


# ── Fixed-key attack (ASCAD_r) ─────────────────────────────────

def attack_fixed_key(preds, plaintexts, true_key, byte_list,
                     n_experiments, traces_per_exp):
    from gmpy2 import mpz, mul as gmpy_mul

    n_traces = preds.shape[0]
    n_bytes = len(byte_list)
    traces_per_exp = min(traces_per_exp, n_traces)

    # Precompute: hyp_s1[trace, byte_idx, key_cand]
    hyp_s1 = np.empty((n_traces, n_bytes, 256), dtype=np.uint8)
    for i, b in enumerate(byte_list):
        for k in range(256):
            hyp_s1[:, i, k] = SBOX[plaintexts[:, b] ^ k]

    history = {}
    for exp in range(n_experiments):
        history[exp] = {"total_rank": []}
        for b in byte_list:
            history[exp][b] = []

        trace_order = np.random.permutation(n_traces)[:traces_per_exp]
        scores = {b: np.zeros(256, dtype=np.float64) for b in byte_list}

        for t_pos, tid in enumerate(trace_order):
            total_rank = mpz(1)
            all_ok = True
            for i, b in enumerate(byte_list):
                for k in range(256):
                    scores[b][k] += np.log(preds[tid, i, hyp_s1[tid, i, k]] + 1e-36)
                rank = _rank_of(scores[b], true_key[b])
                history[exp][b].append(rank)
                total_rank = gmpy_mul(total_rank, mpz(rank))
                if rank > 1:
                    all_ok = False

            ge = _pow_rank(total_rank)
            history[exp]["total_rank"].append(ge)

            if all_ok:
                for rem in range(t_pos + 1, traces_per_exp):
                    for b in byte_list:
                        history[exp][b].append(1)
                    history[exp]["total_rank"].append(0)
                break

        if exp % max(1, n_experiments // 10) == 0:
            print(f"  Exp {exp}: final GE = 2^{history[exp]['total_rank'][-1]}")

    ge_matrix = np.array(
        [[history[i]["total_rank"][j] for j in range(traces_per_exp)]
         for i in range(n_experiments)]
    )
    mean_ge = np.mean(ge_matrix, axis=0)
    ge_lt2 = np.where(mean_ge < 2)[0]
    threshold = int(np.min(ge_lt2)) if len(ge_lt2) > 0 else traces_per_exp

    print(f"\n=== Fixed-key results ({n_experiments} experiments) ===")
    print(f"GE < 2 at trace: {threshold}")
    print(f"Final mean GE: 2^{mean_ge[-1]:.1f}")
    return {"history": history, "mean_ge": mean_ge.tolist(), "ge_lt2": threshold}


# ── Variable-key attack (ASCAD_v2) ─────────────────────────────

def attack_variable_key(preds, plaintexts, true_keys, byte_list):
    n_traces = preds.shape[0]
    n_bytes = len(byte_list)

    ranks = np.empty((n_traces, n_bytes), dtype=np.int32)
    for t in range(n_traces):
        for i, b in enumerate(byte_list):
            scores = np.empty(256, dtype=np.float64)
            for k in range(256):
                scores[k] = np.log(preds[t, i, SBOX[plaintexts[t, b] ^ k]] + 1e-36)
            ranks[t, i] = _rank_of(scores, true_keys[t, b])

    print(f"\n=== Variable-key results ({n_traces} traces) ===")
    for i, b in enumerate(byte_list):
        mean_r = np.mean(ranks[:, i])
        rank1 = np.mean(ranks[:, i] == 1) * 100
        top5 = np.mean(ranks[:, i] <= 5) * 100
        print(f"  Byte {b:2d}: mean rank {mean_r:6.1f} | rank-1 {rank1:5.1f}% | top-5 {top5:5.1f}%")

    overall = np.mean(np.all(ranks == 1, axis=1)) * 100
    print(f"  All bytes rank-1: {overall:.2f}%")
    return {"ranks": ranks.tolist()}


# ── Main ───────────────────────────────────────────────────────

def run_attack(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    cfg = DATASET_CONFIGS[args.dataset]
    byte_list = list(cfg["byte_range"])
    n_bytes = len(byte_list)
    model_dir = cfg["folder"] + "models/"
    metrics_dir = cfg["folder"] + "metrics/"
    os.makedirs(metrics_dir, exist_ok=True)

    # Load attack traces
    print(f"Loading attack traces for {args.dataset}...")
    traces, labels_s1, meta = load_traces_and_labels(
        args.dataset, "attack", n_traces=args.n_traces, target="s1"
    )
    plaintexts = meta["plaintexts"]
    true_keys = meta["keys"]
    n_traces = traces.shape[0]
    trace_length = traces.shape[1]
    print(f"Loaded {n_traces} traces, length {trace_length}")

    # Build & load model
    model = build_model(
        input_length=trace_length,
        byte_range=cfg["byte_range"],
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
        learning_rate=0.001,
        seed=args.seed,
        summary=False,
    )
    weights_path = os.path.join(model_dir, f"{args.model_name}_best.weights.h5")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, f"{args.model_name}_all.weights.h5")
    if not os.path.exists(weights_path):
        print(f"ERROR: No weights found at {weights_path}")
        print("Training may have failed. Skipping attack.")
        return
    print(f"Loading weights from {weights_path}")
    model.load_weights(weights_path)

    # Normalize + predict
    traces_norm = _normalize_traces(traces)[..., np.newaxis]
    print("Running predictions...")
    preds_dict = model.predict({"traces": traces_norm}, batch_size=512, verbose=1)

    # Extract s1 predictions: (n_traces, n_bytes, 256)
    preds = np.empty((n_traces, n_bytes, 256), dtype=np.float32)
    for i, b in enumerate(byte_list):
        preds[:, i, :] = preds_dict[f"output_{b}"]

    # Per-byte accuracy
    print("\n=== Per-byte s1 accuracy ===")
    for i, b in enumerate(byte_list):
        acc = np.mean(np.argmax(preds[:, i], axis=1) == labels_s1[:, b]) * 100
        print(f"  Byte {b:2d}: {acc:.2f}%")

    # Attack
    if cfg["fixed_key"]:
        true_key = list(true_keys[0])
        results = attack_fixed_key(
            preds, plaintexts, true_key, byte_list,
            args.n_experiments, args.traces_per_exp,
        )
    else:
        results = attack_variable_key(preds, plaintexts, true_keys, byte_list)

    # Save
    tag = f"attack_{args.model_name}_{args.n_experiments}exp"
    result_path = os.path.join(metrics_dir, f"{tag}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Key-recovery attack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(__import__('dataset').DATASET_CONFIGS.keys()))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_traces", type=int, default=None)
    parser.add_argument("--n_experiments", type=int, default=100)
    parser.add_argument("--traces_per_exp", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Architecture (must match training)
    parser.add_argument("--convolution_blocks", type=int, default=1)
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=34)
    parser.add_argument("--strides", type=int, default=17)
    parser.add_argument("--pooling_size", type=int, default=2)
    parser.add_argument("--dense_units", type=int, default=200)
    parser.add_argument("--non_shared_blocks", type=int, default=1)
    parser.add_argument("--shared_blocks", type=int, default=1)
    parser.add_argument("--multi_target", action="store_true", default=False)
    parser.add_argument("--n_components", type=int, default=1)
    parser.add_argument("--combiner_rank", type=int, default=64)
    parser.add_argument("--combiner_skip", action="store_true", default=False)
    parser.add_argument("--l2_reg", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--use_attention", action="store_true", default=False)
    parser.add_argument("--attention_heads", type=int, default=4)

    args = parser.parse_args()
    run_attack(args)


if __name__ == "__main__":
    main()
