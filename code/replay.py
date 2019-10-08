import random
from collections import deque
from environment import get_env
import numpy as np


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
        # update = 1 if new_td_error > self.td_error, than the buffer must increase.
        update = (new_td_error - self.td_error) / abs(new_td_error - self.td_error)

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

# TODO: Combined can inherit Naive and overwrite what is necessary. To avoid copy paste bugs.

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

        if len(self.memory) >= self.capacity:
            _ = self.pop()

        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def resize_memory(self, new_size=None):
        """Redefines the size of the buffer.
        Inputs:
            new_size (type: int), capacity = new_size."""

        self.capacity = new_size

        # Oldest experiences are discarded. For Ever.
        # TODO: Check for a more efficient way of cleaning the memory.
        while len(self.memory) > self.capacity:
            _ = self.pop()

    def __len__(self):
        return len(self.memory)

# Add different experience replay methods

class CombinedReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity

        # It is necessary to use List data structure for dynamic buffer.
        self.memory = []

    def pop(self, idx=0):
        # Pop is redefined as taking the oldest element (FIFO) for convinience.
        return self.memory.pop(idx)

    def push(self, transition):

        if len(self.memory) >= self.capacity:
            _ = self.pop()

        self.memory.append(transition)
        self.transition = transition

    def memory_full(self):
        return len(self.memory) >= self.capacity

    def sample(self, batch_size):

        samples = random.sample(self.memory, batch_size - 1)
        samples.append(self.transition)
        return samples

    def __len__(self):
        return len(self.memory)

    def resize_memory(self, new_size=None):
        """Redefines the size of the buffer.
        Inputs:
            new_size (type: int), capacity = new_size."""

        self.capacity = new_size

        # Oldest experiences are discarded. For Ever.
        # TODO: Check for a more efficient way of cleaning the memory.
        while len(self.memory) > self.capacity:
            _ = self.pop()


class SumTree:
    # started from https://github.com/wotmd5731/dqn/blob/master/memory.py
    write = 0

    def __init__(self, max_capacity):
        self.capacity = max_capacity
        self.tree = np.zeros(2 * max_capacity - 1)
        self.data = np.zeros(max_capacity, dtype=object)
        self.num = 0
        self.e = 0.01
        self.a = 0.6

    def _get_priority(self, error):
        if error >= 0:
            return (error + self.e) ** self.a
        else:
            return self._total()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, rand):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if rand <= self.tree[left]:
            return self._retrieve(left, rand)
        else:
            return self._retrieve(right, rand - self.tree[left])

    def _total(self):
        return self.tree[0]

    def add(self, error, data):
        p = self._get_priority(error)
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.num < self.capacity:
            self.num += 1

    def update(self, idx, error):
        p = self._get_priority(error)
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def _get_single(self, a, b):
        rand = random.uniform(a, b)
        idx = self._retrieve(0, rand)
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
            (idx, p, data) = self._get_single(a, b)
            batch.append(data)
            batch_idx.append(idx)
            priorities.append(p)
        return batch, batch_idx, priorities

    def get_len(self):
        return self.num


class RankBased:
    def __init__(self, max_capacity):
        self.capacity = max_capacity
        self.data = deque(maxlen=max_capacity)
        self.priorities = None
        self.total = None
        self.cum_sum = None
        self.update_flag = False
        self.mod = 8
        self.samples_seen = 0
        self.aux = deque(maxlen=max_capacity)

    def add(self, error, data):
        self.samples_seen += 1
        self.aux.append(list(data) + [error])
        if self.samples_seen % self.mod == 0:
            self.data.extend(self.aux)
            self.aux.clear()
            self.update_flag = True
            self.mod = min(10000, self.mod * 2)

    def update(self, idx, error):
        self.data[idx][-1] = error

    def _get_single(self):
        rand = random.uniform(0, self.total)
        index = np.searchsorted(self.cum_sum, rand)
        return index, self.priorities[index], self.data[index][:-1]  # to exclude the error at the end

    def get_batch(self, n):
        if self.update_flag or self.priorities is None:
            self._update_priorities()
        self.total = np.sum(self.priorities)
        self.cum_sum = np.cumsum(self.priorities)
        batch_idx = []
        batch = []
        priorities = []

        for i in range(n):
            (idx, p, data) = self._get_single()
            batch.append(data)
            batch_idx.append(idx)
            priorities.append(p)
        return batch, batch_idx, priorities

    def get_len(self):
        return len(self.data)

    def _update_priorities(self):
        length = self.get_len()
        errors = np.array([data[-1] for data in self.data])
        order = np.argsort(errors)
        order = np.array([order[order[x]] for x in range(length)])
        order = length - order
        self.priorities = 1. / order
        self.update_flag = False


class PrioritizedReplayMemory:  # stored as ( s, a, r, s_ ) in SumTree
    # modified https://github.com/wotmd5731/dqn/blob/master/memory.py

    def __init__(self, max_capacity, method="prop"):
        if method == "prop":
            self.container = SumTree(max_capacity)
        elif method == "rank":
            self.container = RankBased(max_capacity)
        else:
            raise ValueError("Bad replay method")

    def push(self, error, sample):
        self.container.add(error, sample)

    def sample(self, n):
        return self.container.get_batch(n)

    def update(self, idx, error):
        self.container.update(idx, error)

    def __len__(self):
        return self.container.get_len()


# sanity check
if __name__ == "__main__":
    capacity = 10
    memory = PrioritizedReplayMemory(capacity)  # CombinedReplayMemory(capacity)#NaiveReplayMemory(capacity)

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
