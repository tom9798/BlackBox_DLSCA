#!/bin/bash
#SBATCH --job-name=DA_general_gate
#SBATCH --output=/home/projects/sipl-prj10622/GeneralModel/ascadv1/DetailedAttacks/GeneralGate/general_lower_lr_%j.out
#SBATCH --error=/home/projects/sipl-prj10622/GeneralModel/ascadv1/DetailedAttacks/GeneralGate/general_lower_lr_%j.err
#SBATCH --partition=part-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --qos=qos-preempt

# Setup Environment
source /home/tom.soustiel/anaconda3/etc/profile.d/conda.sh
conda activate /home/projects/sipl-prj10622/conda/envs/dlsca_scaaml

# Debug Info
echo "Job running on $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"

cd /home/projects/sipl-prj10622/GeneralModel/ascadv1

# Run Detailed Attack
# Model: general_model_lower_lr (from jobs 6202/6211)
# Training type: general_masking (uses train_models_general.py)
# Arch: resnet, ReduceLR schedule
echo "Detailed attack started at $(date)"
python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/detailed_attack.py \
    --model_name general_model_lower_lr \
    --model_type general_masking \
    --resnet
echo "Detailed attack completed at $(date)"
