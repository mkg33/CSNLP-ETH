#!/bin/bash
# sample sbatch file used in model training
#SBATCH --job-name=OT
#SBATCH --account=csnlp_jobs
#SBATCH --time=24:00:00
#SBATCH --output=max.out


. /etc/profile.d/modules.sh
module add cuda/12.1

source $HOME/torch_env/bin/activate

python3 -u multi0.py \
  --train_features /home/mgwozdz/valid/train_features.pkl \
  --valid_features /home/mgwozdz/valid/valid_features.pkl \
  --out_dir        /home/mgwozdz/sliced/ \
  --epochs 3 --batch_size 32 --n_proj 128 --style_dim 512  --device cuda

python3 -u max_proj_lambda0.py \
  --train_features /home/mgwozdz/valid/train_features.pkl \
  --valid_features /home/mgwozdz/valid/valid_features.pkl \
  --out_dir        /home/mgwozdz/sliced/ \
  --epochs 3 --batch_size 32 --n_proj 128 --style_dim 512  --device cuda

python3 -u sliced_lambda0.py \
  --train_features /home/mgwozdz/valid/train_features.pkl \
  --valid_features /home/mgwozdz/valid/valid_features.pkl \
  --out_dir        /home/mgwozdz/sliced/ \
  --epochs 3 --batch_size 32 --n_proj 128 --style_dim 512  --device cuda
