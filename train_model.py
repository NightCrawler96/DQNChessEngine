import numpy as np
import multiprocessing
import keras
from keras.layers import Dense
import chess_environment.chessboard as cb
from dqn_tools.memory import SimpleMemory
from dqn_tools.trainers import DQNTrainer
from training_tools import DQNChessRecord

# temporary simple model for testing base concept
model = keras.Sequential([
    Dense(256, activation='relu', input_shape=(None, 384)),
    Dense(512, activation='relu'),
    Dense(1, activation='relu')
])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


def choose_action(model: keras.Model, possible_moves, possible_states, fens):
    highest_prize = 0
    best_move = None
    best_state = None
    best_state_fen = None
    for m, s, f in zip(possible_moves, possible_states, fens):
        prize = model.predict(np.array(s).reshape((1, 1, 384)))
        if prize > highest_prize or best_move is None:
            highest_prize = prize
            best_move = m
            best_state = s
            best_state_fen = f

    return best_move, best_state, best_state_fen


def action(acting_model: keras.Model, models_memory: SimpleMemory, environment: cb.ChessBoard, epsilon):
    flip = not environment.current_turn()
    moves, states, fens = environment.get_moves(flip=flip)
    best_move = None
    best_state = None
    best_state_fen = None
    if np.random.uniform(0, 1) < epsilon:
        moves_states_list = list(map(list, zip(moves, states, fens)))
        choices = len(moves_states_list)
        if choices > 1:
            random_element = moves_states_list[np.random.randint(0, choices)]
        else:
            random_element = moves_states_list[0]
        best_move, best_state, best_state_fen = random_element
    else:
        best_move, best_state, best_state_fen = choose_action(acting_model, moves, states, fens)
    # make move
    environment.make_move(best_move, flip)
    real_prize = environment.get_results()
    best_state = np.array(best_state).reshape((1, 384))
    real_prize = np.array([real_prize]).reshape((1, 1))
    if real_prize == cb.IGNORE_GO:
        return
    record = DQNChessRecord()
    record.state = best_state
    record.fen = best_state_fen
    record.reward = real_prize
    models_memory.add(record)


def training(
        acting_model: keras.Model,
        target_model: keras.Model,
        models_memory: SimpleMemory,
        batch_size: int,
        gamma: float):
    training_batch = models_memory.get_batch(batch_size)
    if training_batch is not None:
        samples = [[record.state, record.reward, record.fen] for record in training_batch]
        states, prizes, fens = list(map(list, zip(*samples)))
        reinforced_prizes = []
        for p, f in zip(prizes, fens):
            training_board = cb.ChessBoard(starting_fen=f)
            if not training_board.game_over():
                next_moves, next_states, next_fens = training_board.get_moves()
                _, chosen_state, _ = choose_action(acting_model, next_moves, next_states, next_fens)
                estimated_next_prize = target_model.predict(np.array(chosen_state).reshape((1, 1, 384)))[0]
                reinforced_p = p + gamma * estimated_next_prize
            else:
                reinforced_p = p
            reinforced_prizes.append(reinforced_p)

        states = np.array(states)
        reinforced_prizes = np.array(reinforced_prizes)
        acting_model.train_on_batch(states, reinforced_prizes)


memory = SimpleMemory(int(1e+5))
model_trainer = DQNTrainer(model, memory, action, training)

board = cb.ChessBoard()
TRAINING_STEPS = int(2e+5)
for i in range(TRAINING_STEPS):
    print("Step {} of {}".format(i+1, TRAINING_STEPS))
    model_trainer.take_action(board, 0.3)
    model_trainer.train(batch_size=32, gamma=0.99, theta=0.005)
    if i % 1000 == 0:
        model_trainer.save("./tmp_model.h5")

model_trainer.save("./model.h5")

