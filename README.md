# DLSCA General Model — Usage Guide

This repository contains two model architectures for masking-agnostic Side-Channel Analysis (SCA) of AES power traces. Both target the ASCAD_r dataset (boolean first-order masking, fixed key) and produce Guessing Entropy (GE) as the success metric.

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
- **Trace length**: 700 points per trace
- **Target**: bytes 2–15 of the AES state (14 bytes), `s1 = SBOX[plaintext XOR key]`
- **Masking**: `masked_s1 = s1 XOR mask`, where `mask` is a fresh uniform random byte per trace

---

## Directory Structure

```
DLSCA_GeneralModel/
├── ascadv1/                  # Base model scripts
│   ├── train_models_ResNet.py
│   ├── attack_conf.py
│   └── detailed_attack.py
├── GeneralArch/              # Chosen model (GeneralArch) scripts
│   ├── train.py
│   ├── attack.py
│   └── detailed_attack.py
├── Slurm/
│   ├── ResNet/
│   │   ├── Train&Attack/     # Base model train+attack slurms
│   │   └── DetailedAttack/   # Base model detailed attack slurms
│   └── GeneralArch/
│       ├── Train&Attack/     # Chosen model train+attack slurms
│       └── DetailedAttack/   # Chosen model detailed attack slurms
├── models/                   # Saved model weights (.weights.h5)
├── metrics/                  # Attack result logs
└── Plots/                    # Convergence and attack plots
```

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
