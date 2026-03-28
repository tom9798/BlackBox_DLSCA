#!/bin/bash
#SBATCH --job-name=DA_no_xor_resnet
#SBATCH --output=/home/projects/sipl-prj10622/GeneralModel/ascadv1/DetailedAttacks/ResNet_No_XOR/MTL_shared_mask_no_xor_resnet_%j.out
#SBATCH --error=/home/projects/sipl-prj10622/GeneralModel/ascadv1/DetailedAttacks/ResNet_No_XOR/MTL_shared_mask_no_xor_resnet_%j.err
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
# Model: basic_MTL_hard_sharing_no_xor_with_L2_no_dropout (from job 6203)
# Training type: multi_task_single_target_one_shared_mask_shared_branch_no_xor
# Custom arch: resnet, conv_blocks=2, filters=32, strides=2, pooling=2, dropout=0, L2=0.001
echo "Detailed attack started at $(date)"
python3 /home/projects/sipl-prj10622/GeneralModel/ascadv1/detailed_attack.py \
    --model_name basic_MTL_hard_sharing_no_xor_with_L2_no_dropout \
    --model_type multi_task_single_target_one_shared_mask_shared_branch_no_xor \
    --resnet \
    --convolution_blocks 2 --filters 32 --strides 2 --pooling_size 2 \
    --dropout_rate 0.0 --regularization_factor 0.001
echo "Detailed attack completed at $(date)"
