#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH -c 8

for x in {1..10}
do
    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay PER --env LunarLander-v2 --buffer 10000 --pmethod prop --TAU 0.1 --set_seed x
    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay PER --env LunarLander-v2 --buffer 10000 --pmethod rank --TAU 0.1 --set_seed x
    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay NaiveReplayMemory --env LunarLander-v2 --buffer 10000 --TAU 0.1 --set_seed x
    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay CombinedReplayMemory --env LunarLander-v2 --buffer 10000 --TAU 0.1 --set_seed x

    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.8 --replay PER --pmethod prop --env CartPole-v1 --buffer 10000 --pmethod prop --TAU 0.1 --set_seed x
    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.8 --replay PER --pmethod rank --env CartPole-v1 --buffer 10000 --pmethod prop --TAU 0.1 --set_seed x
    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.8 --replay NaiveReplayMemory --env CartPole-v1 --buffer 10000 --pmethod prop --TAU 0.1 --set_seed x
    python train.py --num_episodes 1000 --batch_size 64 --render_env 0 --num_hidden 64 --lr 5e-4 --discount_factor 0.8 --replay CombinedReplayMemory --env CartPole-v1 --buffer 10000 --pmethod prop --TAU 0.1 --set_seed x

    python train.py --env MountainCar-v0 --lr 5e-4 --render_env 0 --discount_factor 0.99 --TAU 0.1 --buffer 10000 --replay PER --pmethod prop --set_seed x
    python train.py --env MountainCar-v0 --lr 5e-4 --render_env 0 --discount_factor 0.99 --TAU 0.1 --buffer 10000 --replay PER --pmethod rank --set_seed x
    python train.py --env MountainCar-v0 --lr 5e-4 --render_env 0 --discount_factor 0.99 --TAU 0.1 --buffer 10000 --replay NaiveReplayMemory --set_seed x
    python train.py --env MountainCar-v0 --lr 5e-4 --render_env 0 --discount_factor 0.99 --TAU 0.1 --buffer 10000 --replay CombinedReplayMemory --set_seed x
done
