#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=21:00:00              # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=samus.err              # standard error file
#SBATCH --output=samus.out             # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python -m --conf preoperativeSAM.tools.train\
          --task PrePostiUS\
          --model_name SAMUS\
          --dataset_loader pre_and_post\
          --keep_log