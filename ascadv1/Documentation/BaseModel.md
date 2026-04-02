# Multi-Task XOR ResNet — Base Model Documentation

## Overview

A multi-task deep learning model for Side-Channel Analysis (SCA) that extracts AES-128 secret keys from boolean-masked power traces. The model predicts 14 key bytes (bytes 2–15) simultaneously by splitting into two branches: one predicts the masked intermediate value, the other predicts the mask. A hardcoded **XorLayer** combines them to recover the unmasked target.

**Training label:** `t1` (masked intermediate, the AddRoundKey output). The model learns to decompose t1 into its masked component and the mask, then XOR-combines them to predict `s1 = SBOX[plaintext XOR key]`.

**Dataset:** ASCAD_r — boolean-masked AES, 250K-point traces, bytes 2–15.

---

## Model Variants

Three variants are implemented, differing in how the branches share parameters and whether XOR is used:

| Variant | `--training_type` | XOR | Sharing | Result |
|---------|------------------|-----|---------|--------|
| Hard Sharing | `multi_task_single_target_one_shared_mask` | Yes | Mask branch: single shared Dense | GE < 2 at 6 traces |
| Low Sharing (m_d) | `multi_task_single_target_one_shared_mask_shared_branch` | Yes | SharedWeightsDenseLayer (shared W, per-byte bias) | GE < 2 at 7 traces |
| No XOR | `multi_task_single_target_one_shared_mask_shared_branch_no_xor` | No | SharedWeightsDenseLayer + L2 + Dropout | Overfitted (failed) |

---

## Architecture: Hard Sharing

```
                         INPUT TRACE
                       (B, 250000, 1)
                             |
                 +-----------+-----------+
                 |   Adaptive Downsample |   PoolingCrop layers
                 |   (learnable weights) |   halve length until < 25K
                 +-----------+-----------+
                             |
                 +-----------+-----------+
                 |    ResNet Backbone     |   Conv1D(4 filters, k=34, s=17, SELU)
                 |    BN → AvgPool(2)    |   + residual skip (if --resnet)
                 +-----------+-----------+
                             |
                         Flatten
                             |
              +--------------+--------------+
              |                             |
     Mask Branch (shared)        14× Intermediate Branch
     Dense(200, SELU) × 2       Dense(200, SELU) × 2
     Dense(256, Softmax)         Dense(256, no activation)
     → P(mask) (B, 256)         → logits (B, 256) per byte
              |                             |
              +--------+    +---------------+
                       |    |
                  XorLayer per byte
                  P(s1=k) = Σᵢ P(int=i) · P(mask=i⊕k)
                       |
                   Softmax
                       |
                  output_b (B, 256)
```

**Key feature:** The mask branch is a single Dense network whose output is shared identically across all 14 XorLayers. Each byte gets its own intermediate branch (14 independent Dense networks).

---

## Architecture: Low Sharing (Parameter Sharing, m_d design)

```
                         INPUT TRACE
                       (B, 250000, 1)
                             |
                 +-----------+-----------+
                 |   Adaptive Downsample |
                 +-----------+-----------+
                             |
                 +-----------+-----------+
                 |    ResNet Backbone     |
                 +-----------+-----------+
                             |
                         Flatten
                             |
              +--------------+--------------+
              |                             |
     Mask Branch (shared)        Shared Branch (dense_core_shared)
     Dense(200, SELU) × 2       ┌─────────────────────────┐
     Dense(256, Softmax)         │ 14× Dense(200, SELU)    │  non-shared: unique per byte
     → P(mask) (B, 256)         │ → Reshape → (B,200,1)   │
              |                  │ Concatenate → (B,200,14) │
              |                  │                          │
              |                  │ SharedWeightsDenseLayer  │  shared: W shared, bias per byte
              |                  │ (200→200, SELU)          │
              |                  │                          │
              |                  │ SharedWeightsDenseLayer  │  logits layer
              |                  │ (200→256, no activation) │
              |                  │ → (B, 256, 14)           │
              |                  └─────────────────────────┘
              |                             |
              +--------+    +-- slice per byte --+
                       |    |
                  XorLayer per byte
                       |
                   Softmax
                       |
                  output_b (B, 256)
```

**Key difference from Hard Sharing:** The intermediate branches use `SharedWeightsDenseLayer` — a single weight matrix W applied to all 14 bytes, with per-byte bias vectors. This enforces the structural symmetry of AES bytes (all bytes undergo the same S-Box operation) while reducing parameter count.

---

## Custom Layers

### XorLayer

Implements the exact boolean XOR convolution as a fixed (non-trainable) operation:

```
P(s1 = k) = Σᵢ P(intermediate = i) · P(mask = i ⊕ k)
```

Internally:
```python
mapping[i, j] = i ^ j       # 256×256 XOR lookup table
p2 = pred2[:, mapping]      # (B, 256, 256) — gather mask probs by XOR mapping
result = einsum('ij,ijk->ik', pred1, p2)  # (B, 256) — XOR convolution
```

This is mathematically equivalent to computing the probability distribution of the XOR of two random variables given their marginal distributions. It works perfectly for boolean masking but cannot handle other masking schemes (affine, multiplicative).

### SharedWeightsDenseLayer

A Dense layer where the weight matrix W is shared across all bytes, but each byte gets its own bias vector:

```
Input:  (B, input_dim, 14)
Output: (B, units, 14)

For each byte b:
  output[:, :, b] = SELU(W @ input[:, :, b] + bias_b)
```

- **W** shape: `(input_dim, units)` — shared across all 14 bytes
- **bias** shape: `(units, 14)` — one bias vector per byte
- **Parameters:** `input_dim × units + units × 14` (vs `14 × (input_dim × units + units)` if fully independent)

This is the "m_d design" from Marquet & Oswald [1] — it enforces that all bytes use the same learned feature transformation while allowing byte-specific offsets.

### PoolingCrop

Adaptive downsampling for long traces (ASCAD_r has 250K points):

```
For each iteration (until trace_length < 25000):
  x = x * learnable_weights
  x = AveragePooling1D(pool_size=2, strides=2)
  x = BatchNormalization()
  x = AlphaDropout(0.01)
```

Halves the trace length at each step: 250K → 125K → 62.5K → 31.25K → 15625.

---

## ResNet Backbone

When `--resnet` is enabled, the Conv1D block includes a residual skip connection:

```
Main path:
  Conv1D(filters=4, kernel=34, strides=17, activation=SELU)
  → BatchNormalization
  → AveragePooling1D(pool_size=2)

Skip path:
  Conv1D(filters=4, kernel=1, strides=17)  — projection to match dimensions
  → AveragePooling1D(pool_size=2)

Output = Main + Skip
```

Without `--resnet`, only the main path is used (standard CNN).

---

## Training

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--learning_rate` | 0.001 | Adam optimizer LR |
| `--batch_size` | 250 | Traces per batch |
| `--convolution_blocks` | 1 | Number of Conv1D blocks |
| `--filters` | 4 | Conv1D filters per block |
| `--kernel_size` | 34 | Conv1D kernel size |
| `--strides` | 17 | Conv1D stride |
| `--pooling_size` | 2 | AveragePooling pool size |
| `--dropout_rate` | 0.3 | Dropout rate (no_xor variant only) |
| `--regularization_factor` | 1e-4 | L2 weight penalty (no_xor variant only) |
| `--seed` | 42 | Random seed |

### Loss Function

Categorical cross-entropy, computed independently per byte and summed:

```
Total loss = Σ_b  loss_b
loss_b = -log(softmax(output_b)[true_s1_b])
```

All 14 bytes contribute equally. The backbone receives gradients from all bytes simultaneously.

### Training Commands (Successful Models)

**Hard Sharing (no ResNet):**
```bash
python3 train_models_ResNet.py \
    --training_type multi_task_single_target_one_shared_mask \
    --model_name basic_MTL_hard_sharing \
    --seed 42 --epochs 100
```

**Hard Sharing (with ResNet):**
```bash
python3 train_models_ResNet.py \
    --training_type multi_task_single_target_one_shared_mask \
    --model_name basic_MTL_hard_sharing_resnet \
    --seed 42 --epochs 100 --resnet
```

**Low Sharing (no ResNet):**
```bash
python3 train_models_ResNet.py \
    --training_type multi_task_single_target_one_shared_mask_shared_branch \
    --model_name basic_MTL_low_sharing \
    --seed 42 --epochs 100
```

**Low Sharing (with ResNet):**
```bash
python3 train_models_ResNet.py \
    --training_type multi_task_single_target_one_shared_mask_shared_branch \
    --model_name basic_MTL_low_sharing_resnet \
    --seed 42 --epochs 100 --resnet
```

---

## Attack (Guessing Entropy)

The attack uses cumulative log-probability to rank key candidates:

### How it Works

1. **Predict:** Run all attack traces through the trained model → get P(s1_b = v) for each byte b, each trace t, each value v ∈ {0, ..., 255}

2. **Accumulate:** For each candidate key byte k (0–255), for each trace t:
   ```
   score[k] += log(P(s1_b = SBOX[plaintext_t[b] ⊕ k]) + 1e-36)
   ```

3. **Rank:** Sort candidates by score (descending). The rank of the true key byte is its position in this sorted list.

4. **Total GE:** Multiply ranks across all 14 bytes:
   ```
   GE_total = ∏_b rank_b
   GE_bits = log₂(GE_total)
   ```

5. **Report:** GE is reported as 2^x where:
   - **2^0.0** = perfect recovery (all bytes rank #1)
   - **2^7.0** = random guessing
   - **2^93** = no information at all (14 bytes × ~2^6.6 each)

### Attack Results

| Model | ResNet | GE < 2 at trace | Final GE |
|-------|--------|-----------------|----------|
| Hard Sharing | No | 6 | 2^0.0 |
| Hard Sharing | Yes | 11 | 2^0.0 |
| Low Sharing | No | 7 | 2^0.0 |
| Low Sharing | Yes | 8 | 2^0.0 |
| No XOR | Yes | — | 2^90+ (failed) |

All four XOR-based models successfully recover the full key within 6–11 traces.

---

## Limitations

1. **Boolean masking only:** The XorLayer is a fixed operation — it can only handle `masked = s1 ⊕ mask`. For affine masking (`masked = α·s1 + β`), multiplicative masking, or unknown masking schemes, a different combining layer is needed. This limitation is what motivated the GeneralArch model (BilinearCombinerLayer).

2. **Fixed byte range:** Only bytes 2–15 are attacked (14 bytes). Bytes 0–1 are excluded because the ASCAD_r trace window doesn't capture their leakage.

3. **Requires mask branch supervision:** The mask branch predicts P(mask), which implicitly requires the trace to contain mask leakage. If the mask doesn't leak in the power trace (e.g., masked implementation with no intermediate mask storage), the model cannot decompose the prediction.

---

## References

[1] K. Marquet and E. Oswald, "Exploring Multi-Task Learning in the Context of Masked AES Implementations," Springer, 2024.
