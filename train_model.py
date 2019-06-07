import random
import numpy as np
import keras
from keras.layers import Dense
import chess_environment.chessboard as cb

# temporary simple model for testing base concept
model = keras.Sequential([
    Dense(384, activation='relu', input_shape=(None, 384)),
    Dense(20, activation='relu'),
    Dense(1, activation='relu')
])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

memory = []
board = cb.ChessBoard()
for _ in range(100):
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
    memory.append([best_state, real_prize])

    if len(memory) > 32:
        samples = random.sample(memory, k=8)
        samples = list(map(list, zip(*samples)))
        states, prizes = samples
        states = np.array(states)
        prizes = np.array(prizes)
        model.train_on_batch(states, prizes)




