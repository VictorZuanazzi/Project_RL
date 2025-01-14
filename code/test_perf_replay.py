from replay import PrioritizedReplayMemory
import numpy as np
from time import time


def test_performance():
    capacity = 10_000
    num_push = capacity * 2
    num_sample = capacity * 2
    sample_size = 100

    memory = PrioritizedReplayMemory(capacity, method='rank')

    start_push_time = time()
    for _ in range(num_push):
        # Mock transition
        s = np.random.rand(5)
        s_next = np.random.rand(5)
        r = np.random.rand(1)
        a = np.random.randint(0, 5)
        done = False
        err = np.random.rand(1)

        # push transition to memory
        memory.push(err, (s, a, r, s_next, done))
    print(f'push {num_push} took {time() - start_push_time:.3f}s')

    start_sample_time = time()
    for _ in range(num_sample):
        memory.sample(sample_size)
    print(
        f'sample {num_sample} of size {sample_size} took {time() - start_sample_time:.3f}s')

    # reset memory
    memory = PrioritizedReplayMemory(capacity, method='rank')

    start_push_time = time()
    for i in range(num_push):
        # Mock transition
        s = np.random.rand(5)
        s_next = np.random.rand(5)
        r = np.random.rand(1)
        a = np.random.randint(0, 5)
        done = False
        err = np.random.rand(1)

        # push transition to memory
        memory.push(err, (s, a, r, s_next, done))

        n = min(i + 1, sample_size)
        memory.sample(n)
    print(f'mixed {num_push} took {time() - start_push_time:.3f}s')


if __name__ == "__main__":
    test_performance()
