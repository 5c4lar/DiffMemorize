#!/bin/bash

SWEEP_ID="ebumls3r"

# Create sweeps for each of the 4 GPUs
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
done