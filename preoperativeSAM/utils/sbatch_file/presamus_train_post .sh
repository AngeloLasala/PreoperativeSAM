#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=21:00:00              # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=presamus_post.err              # standard error file
#SBATCH --output=presamus_post.out             # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python -m  preoperativeSAM.tools.train\
          --task PrePostiUS\
          --modelname PRESAMUS\
          --dataset_loader post\
          --keep_log