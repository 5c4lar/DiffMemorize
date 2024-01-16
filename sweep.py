# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import wandb

#----------------------------------------------------------------------------
# Define sweep config
sweep_configuration = {
    "program": "train.py",
    "command": ["/usr/bin/env", "torchrun", "--standalone", "--nproc_per_node=1", "train.py", '--outdir=logs', '--cond=0', '--arch=ddpmpp', '--augment=0.0', '--window-size=0.0', '--precond=vp', '--seed=1024', '--duration=200', '--num-blocks=2', '--fp16=True', '--lr=2e-4', '--batch=512', '--wandb_group=width', '${args}'],
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "train_loss"},
    "parameters": {
        "num-channels": {"values": [8, 16, 32, 64, 128, 256, 512]},
        "data": {"values": [1000, 2000, 4000, 8000, 16000, 32000]},
    },
}
sweep_id = wandb.sweep(sweep_configuration, project="DiffusionUnlearning")

print(sweep_id)
#----------------------------------------------------------------------------

# if __name__ == "__main__":
#     wandb.agent(sweep_id)

#----------------------------------------------------------------------------
