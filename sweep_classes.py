# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import wandb
import os

# ----------------------------------------------------------------------------
# Define sweep config
sweep_configuration = {
    "program": "train.py",
    "command": [
        "/usr/bin/env",
        "torchrun",
        "--standalone",
        "--nproc_per_node=auto",
        "train.py",
        "--outdir=logs",
        "--cond=0",
        "--arch=ddpmpp",
        "--augment=0.0",
        "--window-size=0.0",
        "--precond=edm",
        "--seed=1024",
        "--duration=2000",
        "--num-blocks=2",
        "--dtype=bf16",
        "--lr=2e-4",
        "--batch=512",
        "--wandb_group=classes",
        "--num-channels=128",
        "${args}",
    ],
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "train_loss"},
    "parameters": {
        "data": {"values": [os.path.join("datasets/cifar10/data_classes", i) for i in os.listdir("datasets/cifar10/data_classes")]},
    },
}
sweep_id = wandb.sweep(sweep_configuration, project="DiffusionUnlearning")

print(sweep_id)
# ----------------------------------------------------------------------------

# if __name__ == "__main__":
#     wandb.agent(sweep_id)

# ----------------------------------------------------------------------------
