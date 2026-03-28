#!/bin/bash
#SBATCH --job-name=general_lower_lr
#SBATCH --output=/home/projects/sipl-prj10622/GeneralModel/ascadv1/slurm_logs/GeneralGate/Assembling/general_lower_lr_%j.out
#SBATCH --error=/home/projects/sipl-prj10622/GeneralModel/ascadv1/slurm_logs/GeneralGate/Assembling/general_lower_lr_%j.err
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
mkdir -p slurm_logs/Assembling
mkdir -p models
mkdir -p metrics

cd /home/projects/sipl-prj10622/GeneralModel/ascadv1

# Run Training
echo "$(date) CMD: python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/train_models_general.py --training_type general_masking --model_name general_model_lower_lr --seed 42 --epochs 100 --resnet --reduce_lr"
python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/train_models_general.py --training_type general_masking --model_name general_model_lower_lr --seed 42 --epochs 100 --resnet --reduce_lr
echo "Job training completed at $(date)"

echo "$(date) CMD: python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/attack_general.py --model_name general_model_lower_lr --resnet --n_experiments 1000 --n_traces 10000"
python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/attack_general.py --model_name general_model_lower_lr --resnet --n_experiments 1000 --n_traces 10000
echo "Job attack completed at $(date)"