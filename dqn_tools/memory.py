import collections
import os
import pickle
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
        if len(self._deque) < min_rows:
            return None
        batch = random.sample(self._deque, batch_size)
        return batch


class IOMemory:
    def __init__(self,
                 max_size: int,
                 piece_size: int,
                 path: str,
                 saved_pieces: int = 0,
                 min_batches_in_queue: int = 100,
                 min_records: int = 0):
        self._max_len = max_size
        self._piece_size = piece_size
        self._path = path
        self._next_piece = saved_pieces
        self._pieces = saved_pieces
        self._max_pieces = int(max_size / piece_size)
        self._new_memories = []
        self._cashed_memory = []
        self._loaded_piece = None
        self._batches = collections.deque()
        self._min_batches_in_queue = min_batches_in_queue
        self._min_records = min_records

    def add(self, record):
        if len(self._new_memories) >= self._piece_size:
            self._cashed_memory = self._new_memories.copy()
            self.save_piece()
            self._new_memories = []

        self._new_memories.append(record)

    def save_piece(self):
        if not os.path.isdir(self._path):
            os.mkdir(self._path)

        with open("{}/piece_{}.memory".format(self._path, self._next_piece), 'wb') as file:
            pickle.dump(self._cashed_memory, file)

        self._next_piece = (self._next_piece + 1) % self._max_pieces
        if self._pieces < self._max_pieces:
            self._pieces += 1

    def load_piece(self):
        load_piece = random.randint(0, self._pieces - 1)
        with open("{}/piece_{}.memory".format(self._path, load_piece), 'rb') as file:
            self._loaded_piece = pickle.load(file)

    def get_batch(self, batch_size: int):
        if len(self._new_memories) + len(self._cashed_memory) < self._min_records:
            print("No mem!")
            return None

        if len(self._batches) < self._min_batches_in_queue:
            if len(self._cashed_memory) > 0:
                for _ in range(self._min_batches_in_queue):
                    batch = random.sample(self._cashed_memory, batch_size)
                    self._batches.append(batch)
            if self._loaded_piece is not None:
                for _ in range(self._min_batches_in_queue):
                    batch = random.sample(self._loaded_piece, batch_size)
                    self._batches.append(batch)
            if len(self._new_memories) >= batch_size:
                batch = random.sample(self._new_memories, batch_size)
                self._batches.append(batch)

            random.shuffle(self._batches)

        return self._batches.popleft()

    def calculate_size(self):
        return self._pieces * self._piece_size + len(self._new_memories) + len(self._cashed_memory)
