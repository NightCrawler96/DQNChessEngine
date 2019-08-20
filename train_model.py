import numpy as np

import chess_environment.chessboard as cb
from dqn_tools.memory import SimpleMemory
from dqn_tools.trainers import DQNTrainer, load_trainer
from models.BuzdyganDQNv0.template import BuzdyganDQNv0Templte

seed = 12345
np.random.seed(seed)
# temporary simple model for testing base concept
model_template = BuzdyganDQNv0Templte()
LOAD = True
LOAD_FROM = "final/"

if LOAD:
    model_trainer = load_trainer(
        LOAD_FROM,
        "{}_100k".format(model_template.NAME),
        model_template.action,
        model_template.training)
else:
    model = model_template.new_model(seed)
    memory = SimpleMemory(model_template.MEMORY_SIZE)
    model_trainer = DQNTrainer(model, memory, model_template.action, model_template.training)

board = cb.ChessBoard()
for i in range(model_template.START_AT_STEP, model_template.TRAINING_STEPS):
    print("Step {} of {}".format(i+1, model_template.TRAINING_STEPS))
    model_trainer.take_action(board, model_template.get_epsilon(i))
    model_trainer.train(batch_size=model_template.BATCH, gamma=model_template.GAMMA, theta=model_template.THETA)
    if i % model_template.SAVE_PER_STEPS == 0:
        model_trainer.save("tmp", "{}_{}".format(model_template.NAME, i))

model_trainer.save("final", "{}_{}k".format(model_template.NAME,
                                            int(model_template.TRAINING_STEPS / 1000)))

