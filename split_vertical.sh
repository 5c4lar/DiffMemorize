#!/bin/bash
for class in 0 1 2 3 4 5 6 7 8 9
do
    size=5000
    python dataset_utils/dataset_classes.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_classes/cifar10-$class-$size.zip --max-images=$size --classes=$class
done