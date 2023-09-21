import random

from collections import namedtuple

# Memory for Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(
                None
            )  # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = self.Transition(*args)
        self.position = (
            self.position + 1
        ) % self.capacity  # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
