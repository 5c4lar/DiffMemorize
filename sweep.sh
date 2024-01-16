#!/bin/bash

SWEEP_ID="Your Sweep ID"

# Create sweeps for each of the 4 GPUs
# for i in {0..3}
# do
#     CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
# done

CUDA_VISIBLE_DEVICES=0,1,2,3 wandb agent $SWEEP_ID &