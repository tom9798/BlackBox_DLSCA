# DLSCA General Model — Usage Guide

This repository contains two model architectures for masking-agnostic Side-Channel Analysis (SCA) of AES power traces. Both target the ASCAD_r dataset (boolean first-order masking, fixed key) and produce Guessing Entropy (GE) as the success metric.

---

## Environment Setup

**Python version:** 3.10  
**Framework:** TensorFlow 2.18.0 (GPU)  
**CUDA:** 12.x with cuDNN 9.10

### Creating the Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n dlsca python=3.10 -y
conda activate dlsca

# Install dependencies
pip install -r requirements.txt

# Verify GPU is available
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

TensorFlow 2.18.0 will automatically install the required CUDA/cuDNN packages via pip (`nvidia-cudnn-cu12`, `nvidia-cuda-runtime-cu12`, etc.), so a system-level CUDA installation is not required.

### Download and Organize External Data Directory

Run the following once after environment setup. It will:
- download the ASCAD_r dataset file
- download the second Google Drive asset (typically pretrained assets)
- extract it if needed
- enforce the expected directory layout used by `DATASET_CONFIGS`

```bash
# Install downloader helper
pip install gdown

# Choose where to store external data (outside this repository is recommended)
# If your external data root is different (for example a shared mount), set ASCAD_ROOT accordingly
# and update the dataset folder path in GeneralArch/dataset.py to match.
export ASCAD_ROOT="$HOME/ASCAD"
export ASCAD_R_DIR="$ASCAD_ROOT/ASCAD_r"

# Create required directory layout
mkdir -p "$ASCAD_R_DIR/models" "$ASCAD_R_DIR/metrics"

# 1) Main ASCAD_r dataset file
gdown --fuzzy "https://drive.google.com/file/d/1Y44zvuohznNqxaDXht7y4Z1EQJWf1nhQ/view" \
    -O "$ASCAD_R_DIR/Ascad_v1_dataset_full.h5"

# 2) Second shared asset (pretrained weights/metrics bundle)
ASSET_PATH="$ASCAD_R_DIR/pretrained_assets"
gdown --fuzzy "https://drive.google.com/file/d/1I-LLqYpTRuHGXmzFRPDyp8zBrj3GBnUG/view" \
    -O "$ASSET_PATH"

# Extract if archive (zip/tar/tar.gz/tgz). If it's already a file/folder payload, this is skipped.
if file "$ASSET_PATH" | grep -qi 'Zip archive'; then
    unzip -o "$ASSET_PATH" -d "$ASCAD_R_DIR"
elif file "$ASSET_PATH" | grep -Eqi 'tar archive|gzip compressed'; then
    tar -xf "$ASSET_PATH" -C "$ASCAD_R_DIR"
fi

# Normalize possible extracted layouts into ASCAD/ASCAD_r/{models,metrics}
for d in "$ASCAD_R_DIR" "$ASCAD_R_DIR"/*; do
    [ -d "$d/models" ] && cp -rn "$d/models/." "$ASCAD_R_DIR/models/" 2>/dev/null || true
    [ -d "$d/metrics" ] && cp -rn "$d/metrics/." "$ASCAD_R_DIR/metrics/" 2>/dev/null || true
done

# Final expected structure check
echo "Expected paths:"
echo "  $ASCAD_R_DIR/Ascad_v1_dataset_full.h5"
echo "  $ASCAD_R_DIR/models/"
echo "  $ASCAD_R_DIR/metrics/"
ls -lah "$ASCAD_R_DIR"
```

If your external data root is different (for example a shared mount), set `ASCAD_ROOT` accordingly and update the dataset folder path in `GeneralArch/dataset.py` to match.

### GPU Requirements

Training and attack scripts require a CUDA-capable GPU. Experiments were run on NVIDIA A100 (80 GB) and RTX A6000 (48 GB) GPUs. Minimum recommended VRAM:
- Base model: ~4 GB
- GeneralArch (Exp F): ~8 GB
- Large-batch experiments: ~40 GB

---

## Dataset

**ASCAD_r** — Boolean First-Order Masked AES Power Traces

Download: [ASCAD_r on Google Drive](https://drive.google.com/file/d/1Y44zvuohznNqxaDXht7y4Z1EQJWf1nhQ/view)

ASCAD_r contains power traces recorded from an 8-bit ATmega8515 microcontroller executing a first-order boolean-masked AES-128 encryption. At each clock cycle, the device XORs a secret intermediate value with a fresh random mask before operating on it (`masked = s1 XOR mask`), making direct single-trace attacks infeasible.

| Split | Traces | Key |
|-------|--------|-----|
| Training | 50,000 | Variable (25 different keys) |
| Validation | 10,000 | Variable |
| Test (attack) | 5,000 | **Fixed** |

- **Training and validation**: variable keys — used to train and monitor generalization
- **Test (attack)**: fixed key — used for the cumulative log-probability Guessing Entropy attack
- **Trace length**: 250,000 points per trace
- **Target**: bytes 2–15 of the AES state (14 bytes), `s1 = SBOX[plaintext XOR key]`
- **Masking**: `masked_s1 = s1 XOR mask`, where `mask` is a fresh uniform random byte per trace

---

## Project File System Structure

The project consists of two parts: the **code repository** and an external **data directory** where datasets, trained weights, and metrics are stored.

### Code Repository

```
DLSCA_GeneralModel/
│
├── README.md
│
├── ascadv1/                              # Base model (XOR multi-task)
│   ├── train_models_ResNet.py            #   Training script
│   ├── attack_conf.py                    #   GE attack script
│   ├── detailed_attack.py               #   Per-byte rank statistics
│   ├── train_models_general.py           #   GeneralGate training (legacy)
│   ├── attack_general.py                 #   GeneralGate attack
│   ├── utility.py                        #   Data loading helpers
│   ├── utils/                            #   Dataset parameters pickle
│   ├── models/                           #   Symlink or copy of weights
│   ├── metrics/                          #   Training history logs
│   ├── slurm_logs/                       #   SLURM output logs
│   │   ├── ResNet_XOR/                   #     XOR model runs
│   │   ├── ResNet_No_XOR/                #     No-XOR model runs
│   │   └── GeneralGate/Assembling/       #     GeneralGate runs
│   └── DetailedAttacks/                  #   Detailed attack output logs
│       ├── ResNet_XOR/
│       ├── ResNet_No_XOR/
│       └── GeneralGate/
│
├── GeneralArch/                          # Chosen model (BilinearCombinerLayer)
│   ├── train.py                          #   Training script
│   ├── attack.py                         #   GE attack script
│   ├── detailed_attack.py               #   Per-byte rank statistics
│   ├── train_presplit.py                 #   Pre-split combiner diagnostic test
│   ├── model.py                          #   Model architecture (all custom layers)
│   ├── dataset.py                        #   Dataset configs + data loading
│   ├── aes_utils.py                      #   AES S-Box and helpers
│   ├── Documentation/                    #   Architecture documentation
│   ├── models/                           #   Symlink or copy of weights
│   ├── metrics/                          #   Training history logs
│   ├── slurm_logs/                       #   SLURM output logs
│   │   ├── bilinear_v2/                  #     Main experiment logs
│   │   │   └── successfull/              #     Successful run logs
│   │   └── DetailedAttack/               #     Detailed attack logs
│   └── DetailedAttacks/                  #   Detailed attack output
│
├── Slurm/                                # All SLURM job files
│   ├── ResNet/
│   │   ├── Train&Attack/                 #   Base model train+attack slurms
│   │   └── DetailedAttack/               #   Base model detailed attack slurms
│   ├── GeneralGate/
│   │   ├── Train&Attack/
│   │   └── DetailedAttack/
│   └── GeneralArch/
│       ├── Train&Attack/
│       │   └── bilinear_v2/              #   Chosen model experiment slurms
│       └── DetailedAttack/               #   Chosen model detailed attack slurms
│
├── Plots/                                # Convergence and result plots
│   ├── ascadv1/                          #   Base model plots
│   ├── GeneralArch/                      #   Chosen model plots
│   ├── successfull/                      #   All successful run plots
│   └── Overfitted/                       #   All overfitted run plots
│
└── metrics/                              # Top-level metrics
```

### External Data Directory

Datasets and all trained model weights live outside the repository in a shared data directory, due to the weight files exceeding GitHub's file size limit (100 MB). The code references these paths via `DATASET_CONFIGS` in `GeneralArch/dataset.py`.

```
ASCAD/
└── ASCAD_r/                              # Boolean-masked AES dataset
    ├── Ascad_v1_dataset_full.h5          #   250K-point traces, fixed-key attack set
    ├── models/                           #   All trained weights saved here
    │   ├── bv2_allF_best.weights.h5     #     Exp F best checkpoint (GeneralArch)
    │   ├── basic_MTL_hard_sharing_all.weights.h5
    │   ├── basic_MTL_hard_sharing_resnet_all.weights.h5
    │   ├── basic_MTL_low_sharing_all.weights.h5
    │   ├── basic_MTL_low_sharing_resnet_all.weights.h5
    │   └── ...                           #     Other model weights
    └── metrics/                          #   Training histories (.pkl)
```

The `models/` and `metrics/` subdirectories are created automatically during training to store weights and histories.

### HDF5 File Format

Each `.h5` dataset file must contain the following structure:

```
dataset.h5
├── training/
│   ├── traces:      (N_train, trace_length)   — power measurements (int8 or float32)
│   ├── plaintexts:  (N_train, 16)             — AES plaintext bytes
│   ├── keys:        (N_train, 16)             — AES key bytes
│   └── labels/
│       ├── s1:      (N_train, 16)             — SBOX[plaintext XOR key] per byte
│       └── t1:      (N_train, 16)             — optional, for multi-target training
├── validation/  (or test/)
│   └── ...  (same structure)
└── attack/
    └── ...  (same structure)
```

- `s1[i, b] = SBOX[plaintexts[i, b] XOR keys[i, b]]` — the primary training label
- `t1` is an alternative intermediate value, used when `--multi_target` is enabled
- Validation split naming varies: some datasets use `validation/`, others `test/` — configured via `val_split` in `dataset.py`

### Adding a New Dataset

Add an entry to `DATASET_CONFIGS` in `GeneralArch/dataset.py`:

```python
"my_dataset": {
    "folder": "/path/to/dataset/folder/",
    "file": "my_traces.h5",
    "fixed_key_hex": "00112233...",  # hex string if fixed key, None if variable
    "byte_range": range(0, 16),      # which AES bytes to attack (range(2,16) for ASCAD_r)
    "val_split": "validation",       # name of the validation split in the HDF5
    "fixed_key": True,               # True = cumulative attack, False = per-trace attack
},
```

Model weights will be saved to `{folder}/models/` and metrics to `{folder}/metrics/` automatically.

---

## Part 1 — Base Model (Multi-Task XOR ResNet)

### What it does

The base model uses a hardcoded XOR layer to combine two branches:
- **Branch 1** predicts the masked S-Box output
- **Branch 2** predicts the random mask
- An **XorLayer** performs `P(s1 = k) = Σᵢ P(masked=i) · P(mask=i⊕k)` to recover the unmasked target

Two variants are available based on parameter sharing in the branches:

| `--training_type` | Description |
|---|---|
| `multi_task_single_target_one_shared_mask` | Hard sharing: mask branch uses a single shared Dense (no per-byte structure) |
| `multi_task_single_target_one_shared_mask_shared_branch` | Low sharing: both branches use `SharedWeightsDenseLayer` (shared W, per-byte bias) |

### Training

```bash
cd /path/to/DLSCA_GeneralModel/ascadv1

python3 train_models_ResNet.py \
    --training_type multi_task_single_target_one_shared_mask_shared_branch \
    --model_name my_model \
    --seed 42 \
    --epochs 100 \
    --resnet
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--training_type` | `multi_task_single_target_one_shared_mask` | Model variant (see table above) |
| `--model_name` | `model_multi_task_...` | Name used to save weights to `models/` |
| `--seed` | None | Random seed for reproducibility |
| `--epochs` | 100 | Number of training epochs |
| `--resnet` | False | Add residual skip connection in the backbone Conv1D block |
| `--learning_rate` | 0.001 | Adam learning rate |
| `--batch_size` | 250 | Training batch size |
| `--convolution_blocks` | 1 | Number of Conv1D+BN+Pool blocks in the backbone |
| `--filters` | 4 | Number of Conv1D filters per block |
| `--strides` | 17 | Conv1D stride (controls downsampling rate) |
| `--pooling_size` | 2 | AveragePooling1D pool size |
| `--dropout_rate` | 0.3 | Dropout rate (used in regularized variants) |
| `--regularization_factor` | 1e-4 | L2 weight regularization factor |

### Attack (Guessing Entropy)

Loads the best saved weights and runs a cumulative log-probability attack on the ASCAD_r test set. Reports Guessing Entropy per byte.

```bash
python3 attack_conf.py \
    --model_type multi_task_single_target_one_shared_mask_shared_branch \
    --model_name my_model \
    --resnet
```

The `--model_type` must match the `--training_type` used during training. Pass `--resnet` if the model was trained with it.

---

## Part 2 — Base Model Detailed Attack

The detailed attack reports **per-byte rank statistics** using a single trace, showing how close the correct key byte is to the top-1 position without cumulative accumulation.

```bash
cd /path/to/DLSCA_GeneralModel/ascadv1

python3 detailed_attack.py \
    --model_name my_model \
    --model_type multi_task_single_target_one_shared_mask_shared_branch \
    --resnet
```

Output includes:
- Per-byte mean rank, median rank, and top-K percentages across the test set
- Overall attack quality at the single-trace level

---

## Part 3 — Chosen Model: GeneralArch Experiment F

### What it does

Experiment F is the best-performing configuration found during this project. It replaces the hardcoded XorLayer with a **learned BilinearCombinerLayer** (CP decomposition), making the model masking-agnostic — it learns the demasking operation from data instead of having it hardcoded.

**Key architectural differences from the base model:**
1. XorLayer → BilinearCombinerLayer (U, V, T matrices, rank=64)
2. Both branches use SharedWeightsDenseLayer (shared W, per-byte bias), not independent heads
3. Raw logits + LayerNorm feed the combiner (not softmax, which kills gradients)
4. Skip connection: `output = bilinear(A, B) + logits_A` provides gradient path at initialization
5. L2=1e-4 + input noise std=0.1 prevent the skip from becoming a memorization shortcut

**Result:** GE < 2 at **2 traces** on ASCAD_r (best: GE=2^0.0, full recovery in 3–4 traces).

### Experiment F Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ASCAD_r (50K training traces) |
| Backbone | 1 Conv1D block, 4 filters, kernel=34, stride=17, pool=2 |
| Heads | 1 non-shared Dense(200) per byte + 1 SharedWeights block |
| Components | 2 (bilinear combiner) |
| Combiner rank | 64 |
| Skip connection | Yes |
| L2 regularization | 1e-4 |
| Input noise std | 0.1 |
| Learning rate | 0.001 (static) |
| Epochs | 200 |
| Seed | 42 |

### Training

```bash
cd /path/to/DLSCA_GeneralModel/GeneralArch

python3 train.py \
    --dataset ascad_r \
    --model_name bv2_allF \
    --convolution_blocks 1 --filters 4 --kernel_size 34 --strides 17 --pooling_size 2 \
    --dense_units 200 --non_shared_blocks 1 --shared_blocks 1 \
    --n_components 2 --combiner_rank 64 --combiner_skip \
    --l2_reg 0.0001 --noise_std 0.1 \
    --n_traces 50000 \
    --learning_rate 0.001 --static_lr --epochs 200 --seed 42
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | — | Dataset name (e.g. `ascad_r`, `ascad_v2`) |
| `--model_name` | — | Name used to save weights |
| `--convolution_blocks` | 1 | Number of Conv1D backbone blocks |
| `--filters` | 4 | Filters per Conv1D block |
| `--kernel_size` | 34 | Conv1D kernel size |
| `--strides` | 17 | Conv1D stride |
| `--pooling_size` | 2 | AveragePooling size |
| `--dense_units` | 200 | Units in per-byte Dense layers |
| `--non_shared_blocks` | 1 | Number of independent per-byte Dense blocks |
| `--shared_blocks` | 1 | Number of SharedWeightsDenseLayer blocks |
| `--n_components` | 2 | 1 = direct prediction, 2 = bilinear combiner |
| `--combiner_rank` | 64 | CP decomposition rank (R in U, V, T matrices) |
| `--combiner_skip` | False | Add skip: `output += logits_A` |
| `--l2_reg` | 0.0 | L2 weight regularization |
| `--noise_std` | 0.0 | Gaussian noise std added to input traces during training |
| `--n_traces` | all | Number of training traces to use |
| `--learning_rate` | 0.001 | Adam learning rate |
| `--static_lr` | False | Use constant LR (default: ReduceLROnPlateau) |
| `--epochs` | 200 | Number of training epochs |
| `--seed` | 42 | Random seed |
| `--multi_target` | False | Also train on t1 targets for extra gradient signal |

### Attack (Guessing Entropy)

```bash
python3 attack.py \
    --dataset ascad_r \
    --model_name bv2_allF \
    --convolution_blocks 1 --filters 4 --kernel_size 34 --strides 17 --pooling_size 2 \
    --dense_units 200 --non_shared_blocks 1 --shared_blocks 1 \
    --n_components 2 --combiner_rank 64 --combiner_skip \
    --l2_reg 0.0001
```

All architecture flags must match exactly what was used during training, so the model is rebuilt identically before loading weights.

---

## Part 4 — Chosen Model Detailed Attack (Experiment F)

```bash
cd /path/to/DLSCA_GeneralModel/GeneralArch

python3 detailed_attack.py \
    --dataset ascad_r \
    --model_name bv2_allF \
    --convolution_blocks 1 --filters 4 --kernel_size 34 --strides 17 --pooling_size 2 \
    --dense_units 200 --non_shared_blocks 1 --shared_blocks 1 \
    --n_components 2 --combiner_rank 64 --combiner_skip \
    --l2_reg 0.0001
```

Output:
- Per-byte rank of the correct key byte across all test traces (single-trace evaluation)
- Mean rank, median rank, and percentage of traces where the correct byte is in the top 1/3/5
- Overall mean rank across all 14 bytes

---

## Success Metric: Guessing Entropy

GE is reported as **2^x**:
- **GE = 2^0.0** → correct key is always ranked #1 — perfect recovery
- **GE = 2^7.0** → correct key ranks ~128 on average — random (no information)
- **GE < 2 at trace N** → the model correctly identifies the key byte using only N traces

Experiment F achieves **GE < 2 at trace 2** (full key recovery in 3–4 traces).

---

## References

[1] K. Marquet and E. Oswald, "Exploring Multi-Task Learning in the Context of Masked AES Implementations," Springer, 2024.

The base model architecture (Multi-Task XOR ResNet with SharedWeightsDenseLayer) is derived from [1].
