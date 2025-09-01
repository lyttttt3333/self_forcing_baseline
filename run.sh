#!/bin/bash

source /lustre/fsw/portfolios/av/users/shiyil/anaconda3/etc/profile.d/conda.sh
conda activate self_forcing
git pull origin main
torchrun --nnodes=1 --nproc_per_node=6 \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
