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