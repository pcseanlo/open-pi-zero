#!/bin/bash

#SBATCH --job-name=openpi0-eval-bridge
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run \
    scripts/server.py \
    --config-name=bridge \
    --config-path=../config/server \
    device=cuda:0 \
    horizon_steps=4 \
    act_steps=4 \
    use_bf16=False \
    use_torch_compile=True \
    name=bridge_beta \
    'checkpoint_path="bridge_beta_step19296_2024-12-26_22-30_42.pt"'
