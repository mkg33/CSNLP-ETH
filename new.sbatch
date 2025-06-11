#!/bin/bash
#SBATCH --job-name=project
#SBATCH --account=csnlp
#SBATCH --time=08:00:00
#SBATCH --output=contrastive_%j.out


. /etc/profile.d/modules.sh
module add cuda/12.1

source $HOME/torch_env/bin/activate


python3 /home/mgwozdz/cont.py \
    --train-features /home/mgwozdz/valid/train_features.pkl \
    --output-dir     /home/mgwozdz/models/ \
    --device         cuda \
    --epochs         3 \
    --batch-size     32 \
    --lr             3e-5 \
    --proj-dim       768 \
    --queue-size     4096 \
    --num-workers    4 \
    --print-freq     50
