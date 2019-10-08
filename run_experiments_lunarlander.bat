for /l %%x in (1, 1, 1) do (
	echo %%x
	python code/train.py --num_episodes 1000 --batch_size 64 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay CombinedReplayMemory --env LunarLander-v2 --buffer 100000 --pmethod prop --TAU 0.1 --seed_value %%x
	python code/train.py --num_episodes 1000 --batch_size 64 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay NaiveReplayMemory --env LunarLander-v2 --buffer 100000 --pmethod prop --TAU 0.1 --seed_value %%x
	python code/train.py --num_episodes 1000 --batch_size 64 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay PER--env LunarLander-v2 --buffer 100000 --pmethod prop --TAU 0.1 --seed_value %%x
)
PAUSE