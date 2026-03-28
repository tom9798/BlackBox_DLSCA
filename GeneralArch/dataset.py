"""Unified dataset loader for ASCAD_r (boolean), ASCAD_v2 (affine), and unmasked.

Loads HDF5 datasets and returns tf.data.Dataset objects ready for training.
Only uses unmasked labels: s1 and t1.
"""

import numpy as np
import h5py
import tensorflow as tf

# Dataset configurations
DATASET_CONFIGS = {
    "ascad_r": {
        "folder": "/home/projects/sipl-prj10622/ASCAD/ASCAD_r/",
        "file": "Ascad_v1_dataset_full.h5",
        "fixed_key_hex": "00112233445566778899AABBCCDDEEFF",
        "byte_range": range(2, 16),  # bytes 2-15 (14 bytes)
        "val_split": "test",         # validation data lives under "test"
        "fixed_key": True,
    },
    "ascad_v2": {
        "folder": "/home/projects/sipl-prj10622/ASCAD/ASCAD_v2/",
        "file": "Ascad_v2_extracted_exp_experiments.h5",
        "fixed_key_hex": None,       # variable key — read from h5
        "byte_range": range(0, 16),  # all 16 bytes
        "val_split": "validation",   # separate validation split
        "fixed_key": False,
    },
    "unmasked_cw": {
        "folder": "/home/projects/sipl-prj10622/ADDING_UNMASKED_DATA/",
        "file": "unmasked_cw.h5",
        "fixed_key_hex": "2B7E151628AED2A6ABF7158809CF4F3C",
        "byte_range": range(0, 16),  # all 16 bytes
        "val_split": "validation",
        "fixed_key": True,
    },
    "dpa_contest2": {
        "folder": "/home/projects/sipl-prj10622/ASCAD/unmasked_non_ascad/v2 public database/",
        "file": "DPA_contest2_with_s1.h5",
        "fixed_key_hex": None,       # 25 training keys, 4 attack keys
        "byte_range": range(0, 16),  # all 16 bytes
        "val_split": "test",         # use test split for validation
        "fixed_key": False,          # variable key — per-trace attack
    },
    "ascad_v2_full": {
        "folder": "/home/projects/sipl-prj10622/ASCAD/ASCAD_v2_full_extracted/",
        "file": "ascadv2_full_standard.h5",
        "fixed_key_hex": None,       # variable key
        "byte_range": range(0, 16),  # all 16 bytes
        "val_split": "validation",   # 50K val split from conversion
        "fixed_key": False,
    },
    "d1_unprotected": {
        "folder": "/home/projects/sipl-prj10622/AES_PTv2/dataset_1/",
        "file": "D1_unprotected.h5",
        "fixed_key_hex": None,       # variable key
        "byte_range": range(0, 16),  # all 16 bytes
        "val_split": "validation",
        "fixed_key": False,
    },
    "d1_ms1": {
        "folder": "/home/projects/sipl-prj10622/AES_PTv2/dataset_1/",
        "file": "D1_MS1.h5",
        "fixed_key_hex": None,       # variable key, 2-byte boolean mask
        "byte_range": range(0, 16),  # all 16 bytes
        "val_split": "validation",
        "fixed_key": False,
    },
    "d1_ms2": {
        "folder": "/home/projects/sipl-prj10622/AES_PTv2/dataset_1/",
        "file": "D1_MS2.h5",
        "fixed_key_hex": None,       # variable key, 2-byte boolean mask
        "byte_range": range(0, 16),  # all 16 bytes
        "val_split": "validation",
        "fixed_key": False,
    },
}


def _open_h5(dataset_name):
    """Open the HDF5 file for a given dataset."""
    cfg = DATASET_CONFIGS[dataset_name]
    path = cfg["folder"] + cfg["file"]
    return h5py.File(path, "r")


def load_traces_and_labels(
    dataset_name,
    split="training",
    n_traces=None,
    target="s1",
):
    """Load raw traces and labels from HDF5.

    Returns:
        traces: (N, trace_len)
        labels: (N, 16) — target values per byte
        metadata: dict with 'plaintexts' and 'keys' (N, 16)
    """
    f = _open_h5(dataset_name)
    grp = f[split]

    if n_traces is None:
        traces = np.array(grp["traces"])
    else:
        traces = np.array(grp["traces"][:n_traces])

    n = traces.shape[0]
    labels = np.array(grp["labels"][target][:n], dtype=np.uint8)
    plaintexts = np.array(grp["plaintexts"][:n], dtype=np.uint8)
    keys = np.array(grp["keys"][:n], dtype=np.uint8)

    f.close()
    return traces, labels, {"plaintexts": plaintexts, "keys": keys}


def _normalize_traces(traces):
    """Z-score normalize per trace, return float32."""
    mean = np.mean(traces, axis=1, keepdims=True)
    std = np.std(traces, axis=1, keepdims=True) + 1e-8
    return ((traces - mean) / std).astype(np.float32)


def _normalize_batch(x, y):
    """Per-trace z-score normalization applied inside tf.data pipeline."""
    traces = tf.cast(x["traces"], tf.float32)
    mean = tf.reduce_mean(traces, axis=1, keepdims=True)
    std = tf.math.reduce_std(traces, axis=1, keepdims=True) + 1e-8
    x["traces"] = (traces - mean) / std
    return x, y


def _make_augment_fn(noise_std):
    """Return a tf.data .map() function that adds Gaussian noise."""
    def _augment(x, y):
        traces = x["traces"]
        noise = tf.random.normal(tf.shape(traces), stddev=noise_std)
        x["traces"] = traces + noise
        return x, y
    return _augment


def build_tf_dataset(
    dataset_name,
    n_traces=None,
    batch_size=256,
    byte_range=None,
    n_val_traces=10000,
    multi_target=False,
    noise_std=0.0,
):
    """Build training and validation tf.data.Datasets.

    If multi_target=True, labels for both s1 (output_X) and t1 (output_t_X)
    are included to enable multi-target training.

    Traces are kept in their original dtype and normalized per-batch via .map()
    to avoid materializing the entire float32 array (critical for ASCAD_r's
    250K-point traces which would otherwise OOM).

    Returns:
        train_ds, val_ds, trace_length
    """
    cfg = DATASET_CONFIGS[dataset_name]
    if byte_range is None:
        byte_range = cfg["byte_range"]

    # --- Load s1 training and validation ---
    traces_train, s1_train, _ = load_traces_and_labels(
        dataset_name, "training", n_traces, "s1"
    )
    val_split = cfg["val_split"]
    traces_val, s1_val, _ = load_traces_and_labels(
        dataset_name, val_split, n_val_traces, "s1"
    )

    # --- Optionally load t1 for multi-target ---
    if multi_target:
        _, t1_train, _ = load_traces_and_labels(
            dataset_name, "training", n_traces, "t1"
        )
        _, t1_val, _ = load_traces_and_labels(
            dataset_name, val_split, n_val_traces, "t1"
        )

    trace_length = traces_train.shape[1]
    print(f"Trace length: {trace_length}")
    print(f"Training: {traces_train.shape[0]} traces, "
          f"Validation: {traces_val.shape[0]} traces")
    if multi_target:
        print("Multi-target mode: predicting both s1 and t1")

    # Expand channel dim but keep original dtype (int8 for ASCAD_r)
    # Normalization happens per-batch via _normalize_batch in the pipeline
    traces_train = traces_train[..., np.newaxis]
    traces_val = traces_val[..., np.newaxis]

    # Build label dicts
    x_train = {"traces": traces_train}
    y_train = {}
    x_val = {"traces": traces_val}
    y_val = {}

    eye256 = np.eye(256, dtype=np.float32)

    for b in byte_range:
        y_train[f"output_{b}"] = eye256[s1_train[:, b]]
        y_val[f"output_{b}"] = eye256[s1_val[:, b]]

    if multi_target:
        for b in byte_range:
            y_train[f"output_t_{b}"] = eye256[t1_train[:, b]]
            y_val[f"output_t_{b}"] = eye256[t1_val[:, b]]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(traces_train), 50000), reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(_normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    )
    if noise_std > 0:
        train_ds = train_ds.map(
            _make_augment_fn(noise_std), num_parallel_calls=tf.data.AUTOTUNE
        )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .map(_normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, trace_length
