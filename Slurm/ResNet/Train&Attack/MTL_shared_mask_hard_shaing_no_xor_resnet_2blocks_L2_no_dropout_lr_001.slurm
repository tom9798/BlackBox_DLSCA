#!/bin/bash
#SBATCH --job-name=MTL_shared_mask_hard_shaing_no_xor_resnet_2blocks_with_L2_no_dropout_lr_001
#SBATCH --output=/home/projects/sipl-prj10622/GeneralModel/ascadv1/slurm_logs/ResNet_No_XOR/MTL_shared_mask_hard_shaing_no_xor_resnet_2blocks_with_L2_no_dropout_lr_001_%j.out
#SBATCH --error=/home/projects/sipl-prj10622/GeneralModel/ascadv1/slurm_logs/ResNet_No_XOR/MTL_shared_mask_hard_shaing_no_xor_resnet_2blocks_with_L2_no_dropout_lr_001_%j.err
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
mkdir -p slurm_logs
mkdir -p models
mkdir -p metrics

cd /home/projects/sipl-prj10622/GeneralModel/ascadv1

# Run Training
echo "$(date) CMD: python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/train_models_ResNet.py --training_type multi_task_single_target_one_shared_mask_shared_branch_no_xor --model_name basic_MTL_hard_sharing_no_xor_with_L2_no_dropout --seed 42 --epochs 100 --resnet --convolution_blocks 2 --pooling_size 2 --strides 2 --filters 32 --learning_rate 0.0001 --dropout_rate 0.0 --regularization_factor 0.001"
python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/train_models_ResNet.py --training_type multi_task_single_target_one_shared_mask_shared_branch_no_xor --model_name basic_MTL_hard_sharing_no_xor_with_L2_no_dropout --seed 42 --epochs 100 --resnet --convolution_blocks 2 --pooling_size 2 --strides 2 --filters 32 --learning_rate 0.0001 --dropout_rate 0.0 --regularization_factor 0.001 
echo "Job training completed at $(date)"

echo "Job attack started at $(date)"
python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/attack_conf.py --model_type multi_task_single_target_one_shared_mask_shared_branch_no_xor --model_name basic_MTL_hard_sharing_no_xor_with_L2_no_dropout_lr_001
echo "Job attack completed at $(date)"
