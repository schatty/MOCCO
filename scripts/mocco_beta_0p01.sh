#!/bin/bash

for ((i=0; i<10; i+=1))
do
    python train.py --env $1 --algo MOCCO --seed $i --device $2 --beta 0.01
done
