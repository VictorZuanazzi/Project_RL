import random
from collections import deque
from environment import get_env
import numpy as np

import heapq
from itertools import count


class BufferSizeManager:
    def __init__(self, initial_capacity, size_change=20):
        """Adaptive buffer size.

        If size_change > 1:  Linear buffer update as in: https://arxiv.org/pdf/1710.06574.pdf
        If size_change in [0, 1]: Percentage update.
        If size_change < 0 then the algorithm works in the inverse manner as described in the paper.

        You should imagine the buffer manager as a mid-aged fat man that believes his role is key in the success of
        the company, even though many people think they could do without him."""

        self.capacity = initial_capacity
        self.k = size_change
        self.td_error = 0

    def update_td_error(self, new_td_error):
        self.td_error = abs(new_td_error)

    def update_memory_size(self, new_td_error):
        new_td_error = abs(new_td_error)

        # update = -1 if new_td_error < self.td_error, then the buffer must decrease;
        # update = 1 if new_td_error > self.td_error, than the buffer must increase;
        # update = 0 if new_td_error = self.td_error, buffer size remains constant.
        delta = new_td_error - self.td_error
        e = 1e-7

        if abs(delta) < e:
            # for numeric stability
            return self.capacity

        update = delta / abs(delta)

        # allow for non-linear update (not covered in the method proposed by the paper)
        if abs(self.k) < 1:
            update *= int(self.capacity * self.k)
        else:
            update *= int(self.k)

        # Update the buffer size
        self.capacity = max(self.capacity + update, 1)

        # Update the stored td_error
        self.update_td_error(new_td_error)

        return self.capacity

class NaiveReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity

        # List is necessary for dynamic buffer
        self.memory = []  # deque(maxlen=capacity)

    def pop(self, idx=0):
        # Pop is redefined as taking the oldest element (FIFO) for convinience.
        return self.memory.pop(idx)

    def memory_full(self):
        return len(self.memory) >= self.capacity

    def push(self, transition):

        while len(self.memory) >= self.capacity:
            _ = self.pop()

        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def resize_memory(self, new_size=None):
        """Redefines the size of the buffer.
        Inputs:
            new_size (type: int), capacity = new_size."""

        self.capacity = new_size

        # self.push() takes care of decreasing the memory.
        # # Oldest experiences are discarded. For Ever.
        # # TODO: Check for a more efficient way of cleaning the memory.
        # while len(self.memory) > self.capacity:
        #     _ = self.pop()

    def __len__(self):
        return len(self.memory)

# Add different experience replay methods


class CombinedReplayMemory(NaiveReplayMemory):

    def push(self, transition):

        while len(self.memory) >= self.capacity:
            _ = self.pop()

        self.memory.append(transition)
        self.last_transition = transition

    def sample(self, batch_size):

        samples = random.sample(self.memory, batch_size - 1)
        samples.append(self.last_transition)
        return samples


class SumTree:
    # started from https://github.com/wotmd5731/dqn/blob/master/memory.py
    write = 0

    def __init__(self, max_capacity):
        self.capacity = max_capacity
        self.tree = np.zeros(2 * max_capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(max_capacity, dtype=object) # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.num = 0
        self.e = 0.01 # small amount to avoid zero priority
        self.a = 0.6 # [0~1] convert the importance of TD error to priority

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def _propagate_old(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        left = parent * 2 + 1
        right = parent * 2 + 2
        self.tree[parent] = self.tree[right] + self.tree[left]

        if parent != 0:
            self._propagate(parent)

    def _retrieve(self, idx, rand):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # end search when no more child
            return idx

        if rand <= self.tree[left]: # downward search, always search for a higher priority node
            return self._retrieve(left, rand)
        else:
            return self._retrieve(right, rand - self.tree[left])

    def _total(self):
        return self.tree[0] # the root

    def add(self, error, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data # update data_frame

        self.update(idx, error) # update tree_frame

        self.write += 1
        if self.write >= self.capacity: # replace when exceed the capacity
            self.write = 0
        if self.num < self.capacity:
            self.num += 1

    def update(self, idx, error):
        p = self._get_priority(error)
        # change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx)

    def _get_single(self, a, b, rand):
        #rand = random.uniform(a, b)
        idx = self._retrieve(0, rand) # search the max leaf priority based on the lower_bound (rand here)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def get_batch(self, n):
        batch_idx = []
        batch = []
        priorities = []

        segment = self._total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            rand = random.uniform(a, b)
            (idx, p, data) = self._get_single(a, b, rand)
            if data == 0:
                (idx, p, data) = self._get_single(a, b, rand)
            batch.append(data)
            batch_idx.append(idx)
            priorities.append(p)
        if batch[63] == 0:
            batch = batch
        return batch, batch_idx, priorities

    def get_len(self):
        return self.num


class RankBased:
    def __init__(self, max_capacity):
        self.capacity = max_capacity
        self.data = []
        self.priorities = None
        self.total = None
        self.cum_sum = None
        self.tiebreaker = count()

    def memory_full(self):
        return len(self.data) >= self.capacity

    def add(self, error, data):

        # check if there is space left in memory
        while self.memory_full():
            oldest_idx = min(enumerate(self.data), key=lambda d: d[1][1])[0]
            del self.data[oldest_idx]

        # use tie breaker for transitions with equal error
        data = (error, next(self.tiebreaker), *data)
        heapq.heappush(self.data, data)

    def update(self, idx, error):
        self.data[idx] = (error, *self.data[idx][1:])

    def get_batch(self, n):
        self._update_priorities()
        self.total = np.sum(self.priorities)
        self.cum_sum = np.cumsum(self.priorities)

        batch = []
        priorities = []

        # sample hole batch indicies is faster than each individual
        rands = np.random.rand(n) * self.total
        batch_idx = np.searchsorted(self.cum_sum, rands)
        # picking transitions one by one is faster than indixing with a list
        for idx in batch_idx:
            batch.append(self.data[idx][2:])
            priorities.append(self.priorities[idx])

        return batch, batch_idx, priorities

    def get_len(self):
        return len(self.data)

    def _update_priorities(self):
        # order is inverse of actual position in heap
        order = np.array(range(self.get_len() + 1, 1, -1))
        self.priorities = 1. / order


class PrioritizedReplayMemory:
    # stored as ( s, a, r, s_ ) in SumTree
    # modified https://github.com/wotmd5731/dqn/blob/master/memory.py

    def __init__(self, max_capacity, method="prop"):
        if method == "prop":
            self.container = SumTree(max_capacity)
        elif method == "rank":
            self.container = RankBased(max_capacity)
        else:
            raise ValueError("Bad replay method")

    def memory_full(self):
        return self.container.memory_full()

    def push(self, error, sample):
        self.container.add(error, sample)

    def sample(self, n):
        return self.container.get_batch(n)

    def update(self, idx, error):
        self.container.update(idx, error)

    def resize_memory(self, new_size=None):
        """Redefines the size of the buffer.
        Inputs:
            new_size (type: int), capacity = new_size."""

        self.container.capacity = new_size

    def __len__(self):
        return self.container.get_len()


# sanity check
if __name__ == "__main__":
    capacity = 10
    # CombinedReplayMemory(capacity)#NaiveReplayMemory(capacity)
    memory = PrioritizedReplayMemory(capacity)

    env, _ = get_env("Acrobot-v1")

    # Sample a transition
    s = env.reset()
    a = env.action_space.sample()
    s_next, r, done, _ = env.step(a)

    # Push a transition
    err = 0.5
    memory.push(err, (s, a, r, s_next, done))

    # Sample a batch size of 1
    print(memory.sample(1))
