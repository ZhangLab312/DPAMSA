import config
import random


class ReplayMemory:
    def __init__(self):
        self.storage = []
        self.max_size = config.replay_memory_size
        self.size = 0
        self.ptr = 0
        self.previous_hash = None

    def push(self, data: tuple):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr-1] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
            self.ptr += 1
            self.size += 1

    def sample(self, batch_size):
        samples = random.sample(self.storage, batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in range(batch_size):
            s, ns, a, r, d = samples[i]
            state.append(s)
            next_state.append(ns)
            action.append(a)
            reward.append(r)
            done.append(d)

        return state, next_state, action, reward, done
