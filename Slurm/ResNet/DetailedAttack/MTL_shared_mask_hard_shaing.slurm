#!/bin/bash
#SBATCH --job-name=DA_hard_sharing
#SBATCH --output=/home/projects/sipl-prj10622/GeneralModel/ascadv1/DetailedAttacks/ResNet_XOR/MTL_shared_mask_hard_shaing_%j.out
#SBATCH --error=/home/projects/sipl-prj10622/GeneralModel/ascadv1/DetailedAttacks/ResNet_XOR/MTL_shared_mask_hard_shaing_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# Setup Environment
source /home/tom.soustiel/anaconda3/etc/profile.d/conda.sh
conda activate /home/projects/sipl-prj10622/conda/envs/dlsca_scaaml

# Debug Info
echo "Job running on $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"

# Create logs dir
mkdir -p slurm_logs/ResNet_XOR_DetailedAttack

cd /home/projects/sipl-prj10622/GeneralModel/ascadv1

# Run Detailed Attack
echo "Detailed attack started at $(date)"
python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/detailed_attack.py --model_name basic_MTL_hard_sharing --model_type multi_task_single_target_one_shared_mask
echo "Detailed attack completed at $(date)"
