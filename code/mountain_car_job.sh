#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --job-name="mountain_adaptive_buffer"

for x in {1..10}
do
    python train.py \
        --num_episodes 1000 \
        --batch_size 64 \
        --num_hidden 64 \
        --lr 5e-4 \
        --discount_factor 0.99 \
        --render_env 0 \
        --replay PER \
        --pmethod rank \
        --env MountainCar-v0 \
        --buffer 100 \
        --TAU 0.001 \
        --adaptive_buffer \
        --seed_value $x

    python train.py \
        --num_episodes 1000 \
        --batch_size 64 \
        --num_hidden 64 \
        --lr 5e-4 \
        --discount_factor 0.99 \
        --render_env 0 \
        --replay NaiveReplayMemory \
        --env MountainCar-v0 \
        --buffer 100 \
        --TAU 0.001 \
        --adaptive_buffer \
        --seed_value $x

    python train.py \
        --num_episodes 1000 \
        --batch_size 64 \
        --num_hidden 64 \
        --lr 5e-4 \
        --discount_factor 0.99 \
        --render_env 0 \
        --replay CombinedReplayMemory \
        --env MountainCar-v0 \
        --buffer 100 \
        --TAU 0.001 \
        --adaptive_buffer \
        --seed_value $x
done
