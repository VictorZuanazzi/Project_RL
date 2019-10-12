


RUN the train.py files with following default settings.

## CartPole
```
python code/train.py --num_episodes 1000 --batch_size 64 --num_hidden 64 --lr 5e-4 --discount_factor 0.8 --replay NaiveReplayMemory --env CartPole-v1 --buffer 10000 --pmethod prop --TAU 0.001 --seed_value 1

```

## LunarLander

```
python train.py --num_episodes 1000 --batch_size 64 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay NaiveReplayMemory --env LunarLander-v2 --buffer 10000 --pmethod prop --TAU 0.1  --seed_value 1

```

## MountainCar-

```
python code/train.py --num_episodes 1000 --batch_size 64 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay NaiveReplayMemory --env MountainCar-v0 --buffer 10000 --pmethod prop --TAU 0.1 --seed_value 1

```


## plotting:
```
cd code

#### generating the mean and std stats:
python plot.py 
#### generating plots:
python plot.py --common

```
