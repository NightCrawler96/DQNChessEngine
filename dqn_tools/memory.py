import collections
import random


class MemoryRecord:
    def __init__(self):
        self.record = None


class SimpleMemory:
    def __init__(self, size: int):
        self._max_len = size
        self._deque = collections.deque()

    def add(self, record):
        current_size = len(self._deque)
        if current_size >= self._max_len:
            self._deque.popleft()
        self._deque.append(record)

    def get_batch(self, batch_size: int, min_rows: int = None):
        if min_rows is None:
            min_rows = len(self._deque) / 3
        if min_rows < batch_size:
            return None
        batch = random.sample(self._deque, batch_size)
        return batch
