import numpy as np
import keras
from keras.layers import Dense
import chess_environment.chessboard as cb
from dqn_tools.memory import SimpleMemory
from training_tools import DQNChessRecord

# temporary simple model for testing base concept
model = keras.Sequential([
    Dense(384, activation='relu', input_shape=(None, 384)),
    Dense(20, activation='relu'),
    Dense(1, activation='relu')
])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

memory = SimpleMemory(1000)
board = cb.ChessBoard()
TRAINING_STEPS = 256
for i in range(TRAINING_STEPS):
    print("Step {} of {}".format(i+1, TRAINING_STEPS))
    flip = not board.current_turn()
    moves, states = board.get_moves(flip=flip)
    highest_prize = 0
    best_move = None
    best_state = None
    for m, s in zip(moves, states):
        prize = model.predict(np.array(s).reshape((1, 1, 384)))
        if prize > highest_prize or best_move is None:
            highest_prize = prize
            best_move = m
            best_state = s
    # make move
    board.make_move(best_move, flip)
    real_prize = board.get_results()
    best_state = np.array(best_state).reshape((1, 384))
    real_prize = np.array([real_prize]).reshape((1, 1))
    record = DQNChessRecord()
    record.state = best_state
    record.reward = real_prize
    memory.add(record)
    training_batch = memory.get_batch(32)
    if training_batch is not None:
        samples = [[record.state, record.reward] for record in training_batch]
        states, prizes = list(map(list, zip(*samples)))
        states = np.array(states)
        prizes = np.array(prizes)
        model.train_on_batch(states, prizes)
model.save("./model.keras")




