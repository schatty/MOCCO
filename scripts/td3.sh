#!/bin/bash

for ((i=0;i<10;i+=1))
do
    python train.py --env $1 --algo TD3 --seed $i --device $2
done