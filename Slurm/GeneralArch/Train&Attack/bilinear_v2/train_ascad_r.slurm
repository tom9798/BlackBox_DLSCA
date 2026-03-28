#!/bin/bash
#SBATCH --job-name=bv2_ascad_r
#SBATCH --output=/home/projects/sipl-prj10622/GeneralModel/GeneralArch/slurm_logs/ascad_r_%j.out
#SBATCH --error=/home/projects/sipl-prj10622/GeneralModel/GeneralArch/slurm_logs/ascad_r_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

source /home/tom.soustiel/anaconda3/etc/profile.d/conda.sh
conda activate /home/projects/sipl-prj10622/conda/envs/dlsca_scaaml
cd /home/projects/sipl-prj10622/GeneralModel/GeneralArch

echo "Job running on $(hostname) | $(date)"

mkdir -p slurm_logs/bilinear_v2

TRAIN=train.py
ATTACK=attack.py

# Backbone matching the WORKING XorLayer model exactly
BACKBONE="--convolution_blocks 1 --filters 4 --kernel_size 34 --strides 17 --pooling_size 2"
HEADS="--dense_units 200 --non_shared_blocks 1 --shared_blocks 1"

# =============================================================================
# Exp A: Bilinear on LOGITS (no softmax) — the gradient fix
#   Same backbone as working XorLayer model, rank=64, LR=0.001
# =============================================================================
echo "$(date) === Exp A: logit-bilinear rank=64, LR=0.001 ==="
python3 $TRAIN --dataset ascad_r --model_name bv2_r64_A \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 \
    --n_traces 50000 \
    --learning_rate 0.001 --static_lr --epochs 200 --seed 42

echo "$(date) Attack: bv2_r64_A"
python3 $ATTACK --dataset ascad_r --model_name bv2_r64_A \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64

# =============================================================================
# Exp B: Bilinear + skip connection
#   output = bilinear(A, B) + A — direct gradient path for branch A
# =============================================================================
echo "$(date) === Exp B: bilinear + skip, rank=64, LR=0.001 ==="
python3 $TRAIN --dataset ascad_r --model_name bv2_skip_B \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 --combiner_skip \
    --n_traces 50000 \
    --learning_rate 0.001 --static_lr --epochs 200 --seed 42

echo "$(date) Attack: bv2_skip_B"
python3 $ATTACK --dataset ascad_r --model_name bv2_skip_B \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 --combiner_skip

# =============================================================================
# Exp C: Higher rank=256 (can exactly represent XOR)
# =============================================================================
echo "$(date) === Exp C: rank=256, LR=0.001 ==="
python3 $TRAIN --dataset ascad_r --model_name bv2_r256_C \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 256 \
    --n_traces 50000 \
    --learning_rate 0.001 --static_lr --epochs 200 --seed 42

echo "$(date) Attack: bv2_r256_C"
python3 $ATTACK --dataset ascad_r --model_name bv2_r256_C \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 256

# =============================================================================
# Exp D: Lower LR=0.0001, longer training (300ep)
# =============================================================================
echo "$(date) === Exp D: rank=64, LR=0.0001, 300ep ==="
python3 $TRAIN --dataset ascad_r --model_name bv2_slow_D \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 \
    --n_traces 50000 \
    --learning_rate 0.0001 --static_lr --epochs 300 --seed 42

echo "$(date) Attack: bv2_slow_D"
python3 $ATTACK --dataset ascad_r --model_name bv2_slow_D \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64

# =============================================================================
# Exp E: Deep heads, no combiner (n_components=1, shared_blocks=4)
#   Can a deep shared network learn demasking implicitly?
# =============================================================================
echo "$(date) === Exp E: direct, deep heads (shared=4), LR=0.001 ==="
python3 $TRAIN --dataset ascad_r --model_name bv2_deep_E \
    $BACKBONE \
    --dense_units 200 --non_shared_blocks 1 --shared_blocks 4 \
    --n_components 1 \
    --n_traces 50000 \
    --learning_rate 0.001 --static_lr --epochs 200 --seed 42

echo "$(date) Attack: bv2_deep_E"
python3 $ATTACK --dataset ascad_r --model_name bv2_deep_E \
    $BACKBONE \
    --dense_units 200 --non_shared_blocks 1 --shared_blocks 4 \
    --n_components 1

# =============================================================================
# Exp F: rank=64 + skip + L2 + noise — all regularization
# =============================================================================
echo "$(date) === Exp F: bilinear+skip+reg, LR=0.001 ==="
python3 $TRAIN --dataset ascad_r --model_name bv2_allF \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 --combiner_skip \
    --l2_reg 0.0001 --noise_std 0.1 \
    --n_traces 50000 \
    --learning_rate 0.001 --static_lr --epochs 200 --seed 42

echo "$(date) Attack: bv2_allF"
python3 $ATTACK --dataset ascad_r --model_name bv2_allF \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 --combiner_skip \
    --l2_reg 0.0001

# =============================================================================
# Exp G: Seed robustness (seed=7)
# =============================================================================
echo "$(date) === Exp G: rank=64, seed=7 ==="
python3 $TRAIN --dataset ascad_r --model_name bv2_seed7_G \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 \
    --n_traces 50000 \
    --learning_rate 0.001 --static_lr --epochs 200 --seed 7

echo "$(date) Attack: bv2_seed7_G"
python3 $ATTACK --dataset ascad_r --model_name bv2_seed7_G \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 --seed 7

echo "$(date) All experiments completed."
