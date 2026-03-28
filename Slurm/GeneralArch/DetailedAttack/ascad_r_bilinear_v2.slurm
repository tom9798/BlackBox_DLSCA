#!/bin/bash
#SBATCH --job-name=DA_genarch_r
#SBATCH --output=/home/projects/sipl-prj10622/GeneralModel/GeneralArch/slurm_logs/DetailedAttack/ascad_r_bilinear_v2_%j.out
#SBATCH --error=/home/projects/sipl-prj10622/GeneralModel/GeneralArch/slurm_logs/DetailedAttack/ascad_r_bilinear_v2_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source /home/tom.soustiel/anaconda3/etc/profile.d/conda.sh
conda activate /home/projects/sipl-prj10622/conda/envs/dlsca_scaaml

echo "Job running on $(hostname) | $(date)"

cd /home/projects/sipl-prj10622/GeneralModel/GeneralArch

DA=detailed_attack.py
BACKBONE="--convolution_blocks 1 --filters 4 --kernel_size 34 --strides 17 --pooling_size 2"
HEADS="--dense_units 200 --non_shared_blocks 1 --shared_blocks 1"

# =============================================================================
# ASCAD_r — Experiment F (bilinear+skip+L2+noise, the winner)
# =============================================================================

echo "$(date) === DA Exp F: bv2_allF ==="
python3 $DA --dataset ascad_r --model_name bv2_allF \
    $BACKBONE $HEADS \
    --n_components 2 --combiner_rank 64 --combiner_skip \
    --l2_reg 0.0001

echo "$(date) Detailed attack completed."
