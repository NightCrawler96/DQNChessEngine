import random

from dqn_tools.memory import IOMemory

memory = IOMemory(50000, 10000, "./mem_test")

for _ in range(5000):
    memory.add(random.randint(0, 100))

for _ in range(150):
    memory.get_batch(32)

for _ in range(6000):
    memory.add(random.randint(0, 100))

for _ in range(150):
    memory.get_batch(32)

for _ in range(5000):
    memory.add(random.randint(0, 100))

for _ in range(150):
    memory.get_batch(32)

memory.load_piece()

for _ in range(1000):
    memory.add(random.randint(0, 100))

for _ in range(150):
    memory.get_batch(32)

