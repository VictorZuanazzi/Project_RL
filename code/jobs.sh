#!/bin/sh

for x in {1..10}
do
    python train.py \
        --num_episodes 1000 \
        --batch_size 64 \
        --num_hidden 64 \
        --lr 5e-4 \
        --discount_factor 0.8 \
        --replay PER \
        --env CartPole-v1 \
        --buffer 10000 \
        --pmethod rank \
        --TAU 0.001 \
        --seed_value $x
done
