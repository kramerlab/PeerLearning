import random
from collections import deque


class SuggestionBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        if len(self.buffer) > batch_size:
            return random.sample(self.buffer, batch_size)
        # else return None

    def latest(self):
        return [self.buffer[-1]]


if __name__ == "__main__":
    buffer = SuggestionBuffer(50)

    for i in range(100):
        buffer.add(i, i % 9, i % 4, {})

    samples = buffer.sample(3)

    print(samples)
