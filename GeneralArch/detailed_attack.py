"""Detailed per-byte rank analysis for a trained model.

Prints rank of the correct key byte at each trace, showing whether
the correct key is consistently in top-N even when not top-1.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from dataset import load_traces_and_labels, DATASET_CONFIGS, _normalize_traces
from model import build_model
from aes_utils import SBOX


def detailed_attack(args):
    np.random.seed(42)
    tf.random.set_seed(42)

    cfg = DATASET_CONFIGS[args["dataset"]]
    byte_list = list(cfg["byte_range"])
    n_bytes = len(byte_list)
    model_dir = cfg["folder"] + "models/"

    # Load attack traces
    traces, labels_s1, meta = load_traces_and_labels(
        args["dataset"], "attack", n_traces=None, target="s1"
    )
    plaintexts = meta["plaintexts"]
    true_keys = meta["keys"]
    n_traces = traces.shape[0]
    trace_length = traces.shape[1]
    print(f"Loaded {n_traces} attack traces, length {trace_length}")

    # Build & load model
    model = build_model(
        input_length=trace_length,
        byte_range=cfg["byte_range"],
        **args["model_kwargs"],
        summary=False,
    )

    weights_path = os.path.join(model_dir, f"{args['model_name']}_best.weights.h5")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, f"{args['model_name']}_all.weights.h5")
    print(f"Loading weights: {weights_path}")
    model.load_weights(weights_path)

    # Predict
    traces_norm = _normalize_traces(traces)[..., np.newaxis]
    preds_dict = model.predict({"traces": traces_norm}, batch_size=512, verbose=0)

    preds = np.empty((n_traces, n_bytes, 256), dtype=np.float32)
    for i, b in enumerate(byte_list):
        preds[:, i, :] = preds_dict[f"output_{b}"]

    # --- Per-trace, per-byte rank of the correct key ---
    fixed_key = cfg.get("fixed_key", True)
    if fixed_key:
        true_key = true_keys[0]
        print(f"\nTrue key (fixed): {true_key.tolist()}")
    else:
        print(f"\nVariable-key dataset ({n_traces} different keys)")

    # For each trace, compute key scores and rank the correct key
    all_ranks = np.empty((n_traces, n_bytes), dtype=np.int32)
    for t in range(n_traces):
        tk = true_keys[0] if fixed_key else true_keys[t]
        for i, b in enumerate(byte_list):
            scores = np.zeros(256, dtype=np.float64)
            for k in range(256):
                scores[k] = preds[t, i, SBOX[plaintexts[t, b] ^ k]]
            sorted_keys = np.argsort(scores)[::-1]
            all_ranks[t, i] = int(np.where(sorted_keys == tk[b])[0][0]) + 1

    # --- Print per-byte statistics ---
    print("\n=== Per-byte rank statistics (single-trace) ===")
    print(f"{'Byte':>6} {'Mean':>8} {'Median':>8} {'Top-1%':>8} {'Top-3%':>8} {'Top-5%':>8} {'Top-10%':>8} {'Top-20%':>8}")
    for i, b in enumerate(byte_list):
        ranks = all_ranks[:, i]
        mean_r = np.mean(ranks)
        med_r = np.median(ranks)
        top1 = np.mean(ranks == 1) * 100
        top3 = np.mean(ranks <= 3) * 100
        top5 = np.mean(ranks <= 5) * 100
        top10 = np.mean(ranks <= 10) * 100
        top20 = np.mean(ranks <= 20) * 100
        print(f"{b:>6} {mean_r:>8.1f} {med_r:>8.1f} {top1:>8.1f} {top3:>8.1f} {top5:>8.1f} {top10:>8.1f} {top20:>8.1f}")

    # Overall
    mean_all = np.mean(all_ranks)
    print(f"\n  Overall mean rank: {mean_all:.1f}")
    print(f"  All bytes top-1 on same trace: {np.mean(np.all(all_ranks == 1, axis=1)) * 100:.2f}%")

    # --- Cumulative log-prob attack (fixed key only) ---
    if fixed_key:
        true_key = true_keys[0]
        print("\n=== Cumulative log-probability attack ===")
        n_exp = 100
        traces_per_exp = min(100, n_traces)

        for exp_id in range(min(5, n_exp)):  # Print first 5 experiments in detail
            trace_order = np.random.permutation(n_traces)[:traces_per_exp]
            scores = {b: np.zeros(256, dtype=np.float64) for b in byte_list}
            print(f"\n  Experiment {exp_id}:")

            for t_pos, tid in enumerate(trace_order[:20]):  # First 20 traces
                byte_ranks = []
                for i, b in enumerate(byte_list):
                    for k in range(256):
                        scores[b][k] += np.log(preds[tid, i, SBOX[plaintexts[tid, b] ^ k]] + 1e-36)
                    sorted_keys = np.argsort(scores[b])[::-1]
                    rank = int(np.where(sorted_keys == true_key[b])[0][0]) + 1
                    byte_ranks.append(rank)

                max_rank = max(byte_ranks)
                mean_rank = np.mean(byte_ranks)
                all_top1 = all(r == 1 for r in byte_ranks)
                status = "ALL TOP-1" if all_top1 else f"max={max_rank}"
                print(f"    Trace {t_pos+1:3d}: ranks={byte_ranks} | mean={mean_rank:.1f} | {status}")

                if all_top1:
                    print(f"    >>> ALL BYTES RECOVERED at trace {t_pos+1}!")
                    break
    else:
        print("\n(Cumulative attack skipped — variable-key dataset)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detailed per-byte rank analysis for GeneralArch models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(__import__('dataset').DATASET_CONFIGS.keys()))
    parser.add_argument("--model_name", type=str, required=True)

    # Architecture flags (must match the trained model)
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
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    detailed_attack({
        "dataset": args.dataset,
        "model_name": args.model_name,
        "model_kwargs": {
            "convolution_blocks": args.convolution_blocks,
            "filters": args.filters,
            "kernel_size": args.kernel_size,
            "strides": args.strides,
            "pooling_size": args.pooling_size,
            "dense_units": args.dense_units,
            "non_shared_blocks": args.non_shared_blocks,
            "shared_blocks": args.shared_blocks,
            "n_components": args.n_components,
            "combiner_rank": args.combiner_rank,
            "combiner_skip": args.combiner_skip,
            "l2_reg": args.l2_reg,
            "dropout_rate": args.dropout_rate,
            "use_attention": args.use_attention,
            "attention_heads": args.attention_heads,
            "multi_target": args.multi_target,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    })
