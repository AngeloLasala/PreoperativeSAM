#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=21:00:00              # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=autosamus_post.err              # standard error file
#SBATCH --output=autosamus_post.out             # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python -m  preoperativeSAM.tools.train\
          --task PrePostiUS\
          --modelname AutoSAMUS\
          --dataset_loader post\
          --keep_log