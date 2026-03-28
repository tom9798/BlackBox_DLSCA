"""Detailed per-byte rank analysis for trained ascadv1 models.

Prints rank of the correct key byte at each trace, showing whether
the correct key is consistently in top-N even when not top-1.

Adapted from GeneralArch/detailed_attack.py for the ascadv1 model infrastructure.
"""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utility import (
    read_from_h5_file,
    get_hot_encode,
    load_model_from_name,
    get_rank,
    get_pow_rank,
    XorLayer,
    InvSboxLayer,
    METRICS_FOLDER,
)


def detailed_attack(model_name, model_builder, n_traces=None, resnet=False, **builder_kwargs):
    np.random.seed(42)
    tf.random.set_seed(42)

    byte_list = list(range(2, 16))
    n_bytes = len(byte_list)

    # Load attack traces
    traces, labels_dict, metadata = read_from_h5_file(
        n_traces=n_traces or 10000, dataset="attack", load_plaintexts=True
    )
    n_traces_loaded = traces.shape[0]
    input_length = traces.shape[1]
    print(f"Loaded {n_traces_loaded} attack traces, length {input_length}")

    # Build & load model
    model = model_builder(input_length=input_length, resnet=resnet, **builder_kwargs)
    weights_name = f"{model_name}_all.weights.h5"
    model = load_model_from_name(model, weights_name)

    # Predict
    powervalues = np.expand_dims(traces, 2)
    predictions_raw = model.predict({"traces": powervalues}, verbose=0)

    # Convert predictions to key-probabilities (XOR with plaintexts)
    plaintexts = np.array(metadata["plaintexts"], dtype=np.uint8)[:n_traces_loaded]
    plaintexts_hot = get_hot_encode(plaintexts)

    xor_op = XorLayer(name="xor")
    predictions = np.empty((n_traces_loaded, n_bytes, 256), dtype=np.float32)

    batch_size = 1000
    for batch in range(0, n_traces_loaded // batch_size):
        for byte in range(2, 16):
            predictions[batch * batch_size:(batch + 1) * batch_size, byte - 2] = xor_op(
                [
                    predictions_raw[f"output_{byte}"][batch * batch_size:(batch + 1) * batch_size],
                    plaintexts_hot[batch * batch_size:(batch + 1) * batch_size, byte],
                ]
            )

    # Fixed master key
    master_key = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                  0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]
    true_key = master_key

    print(f"\nTrue key: {true_key}")

    # --- Per-trace, per-byte rank of the correct key ---
    all_ranks = np.empty((n_traces_loaded, n_bytes), dtype=np.int32)
    for t in range(n_traces_loaded):
        for i, b in enumerate(byte_list):
            sorted_keys = np.argsort(predictions[t, i])[::-1]
            all_ranks[t, i] = int(np.where(sorted_keys == true_key[b])[0][0]) + 1

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

    # --- Cumulative log-prob attack (like the standard attack) ---
    print("\n=== Cumulative log-probability attack ===")
    n_exp = 100
    traces_per_exp = min(100, n_traces_loaded)

    for exp_id in range(min(5, n_exp)):  # Print first 5 experiments in detail
        trace_order = np.random.permutation(n_traces_loaded)[:traces_per_exp]
        scores = {b: np.zeros(256, dtype=np.float64) for b in byte_list}
        print(f"\n  Experiment {exp_id}:")

        for t_pos, tid in enumerate(trace_order[:20]):  # First 20 traces
            byte_ranks = []
            for i, b in enumerate(byte_list):
                scores[b] += np.log(predictions[tid, i] + 1e-36)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detailed per-byte rank analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model basename")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model architecture (general_masking, or any ResNet training_type from attack_conf)")
    parser.add_argument("--n_traces", type=int, default=10000, help="Number of attack traces")
    parser.add_argument("--resnet", action="store_true", default=False, help="Use ResNet variant")
    # Optional model kwargs (for non-default architectures like no_xor)
    parser.add_argument("--convolution_blocks", type=int, default=1)
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--strides", type=int, default=5)
    parser.add_argument("--pooling_size", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--regularization_factor", type=float, default=1e-4)
    args = parser.parse_args()

    if args.model_type == "general_masking":
        from train_models_general import general_masking
        builder = general_masking
    else:
        from train_models_ResNet import (
            model_multi_task_single_target,
            model_single_task,
            model_multi_task_single_target_one_shared_mask,
            model_multi_task_single_target_not_shared,
            model_multi_task_single_target_one_shared_mask_shared_branch,
            model_multi_task_affine,
            model_multi_task_single_target_one_shared_mask_shared_branch_general_masking,
            model_multi_task_single_target_one_shared_mask_shared_branch_no_xor,
        )
        resnet_builders = {
            "multi_task_single_target": model_multi_task_single_target,
            "multi_task_single_target_not_shared": model_multi_task_single_target_not_shared,
            "multi_task_single_target_one_shared_mask": model_multi_task_single_target_one_shared_mask,
            "multi_task_single_target_one_shared_mask_shared_branch": model_multi_task_single_target_one_shared_mask_shared_branch,
            "multi_task_single_target_one_shared_mask_shared_branch_general_masking": model_multi_task_single_target_one_shared_mask_shared_branch_general_masking,
            "multi_task_single_target_one_shared_mask_shared_branch_no_xor": model_multi_task_single_target_one_shared_mask_shared_branch_no_xor,
        }
        if args.model_type not in resnet_builders:
            raise ValueError(f"Unknown model_type '{args.model_type}'. "
                             f"Choose from: general_masking, {', '.join(resnet_builders.keys())}")
        builder = resnet_builders[args.model_type]

    # Only pass extra kwargs for model types that accept them
    extra_kwargs = {}
    if args.model_type in ("multi_task_single_target_one_shared_mask_shared_branch_no_xor",):
        extra_kwargs = {
            "convolution_blocks": args.convolution_blocks,
            "filters": args.filters,
            "strides": args.strides,
            "pooling_size": args.pooling_size,
            "dropout_rate": args.dropout_rate,
            "regularization_factor": args.regularization_factor,
        }

    detailed_attack(
        model_name=args.model_name,
        model_builder=builder,
        n_traces=args.n_traces,
        resnet=args.resnet,
        **extra_kwargs,
    )
