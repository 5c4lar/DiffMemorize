#!/bin/bash
shards=10
size=$((50000 / $shards))
for shard in $(seq 0 $(($shards - 1)))
do
    python dataset_utils/dataset_horizontal.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_slices_${shards}/cifar10-$shard.zip --max-images=$size --idx=$shard
done