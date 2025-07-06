'''
Author: David Megli
Date: 2025-05-02
File: replay_buffer.py
Description: Replay Buffer implementation
'''
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch.state, batch.action, batch.reward, batch.next_state, batch.done

    def __len__(self):
        return len(self.buffer)
