import numpy as np
import keras
from keras.initializers import RandomNormal
from keras.layers import Dense, LeakyReLU
from keras.regularizers import l2

import chess_environment.chessboard as cb
from dqn_tools.memory import SimpleMemory
from dqn_tools.trainers import DQNTrainer, load_trainer
from training_tools import DQNChessRecord

seed = 12345
np.random.seed(seed)
# temporary simple model for testing base concept
weight_decay = l2(1e-2)
weight_initializer = RandomNormal(mean=0., stddev=0.02, seed=seed)
model = keras.Sequential([
    Dense(150, input_shape=(384,),
          kernel_initializer=weight_initializer,
          bias_initializer=weight_initializer,
          kernel_regularizer=weight_decay,
          bias_regularizer=weight_decay),
    LeakyReLU(alpha=0.3),
    Dense(300,
          kernel_initializer=weight_initializer,
          bias_initializer=weight_initializer,
          kernel_regularizer=weight_decay,
          bias_regularizer=weight_decay),
    LeakyReLU(alpha=0.3),
    Dense(1, activation="linear",
          kernel_initializer=weight_initializer,
          bias_initializer=weight_initializer,
          kernel_regularizer=weight_decay,
          bias_regularizer=weight_decay)
])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

NAME = "LeakyDQNv0_95000"
LOAD = True
LOAD_FROM = "./tmp"
START_AT_STEP = 95001
TRAINING_STEPS = int(12e+4)
MEMORY_SIZE = int(15e+4)
START_TRAINING_AT = 1000
BATCH = 64
GAMMA = 0.99
THETA = 0.01
EPSILON = 0.2
EPSILON_TRESHOLD = START_TRAINING_AT * 1.1


def get_epsilon(step: int):
    return 1 if step < EPSILON_TRESHOLD else EPSILON


def choose_action(model: keras.Model, possible_moves, possible_states, fens):
    highest_prize = 0
    best_move = None
    best_state = None
    best_state_fen = None
    for m, s, f in zip(possible_moves, possible_states, fens):
        prize = model.predict(np.array(s).reshape((1, 384)))
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
    real_prize = environment.get_reward()
    best_state = np.array(best_state).reshape((384,))
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
    training_batch = models_memory.get_batch(batch_size, min_rows=START_TRAINING_AT)
    if training_batch is not None:
        samples = [[record.state, record.reward, record.fen] for record in training_batch]
        states, prizes, fens = list(map(list, zip(*samples)))
        reinforced_prizes = []
        for p, f in zip(prizes, fens):
            training_board = cb.ChessBoard(starting_fen=f)
            p = p[0]
            if not training_board.game_over():
                next_moves, next_states, next_fens = training_board.get_moves()
                _, chosen_state, _ = choose_action(acting_model, next_moves, np.array(next_states), next_fens)
                estimated_next_prize = target_model.predict(np.array(chosen_state.reshape((1, 384))))[0]
                reinforced_p = p + gamma * estimated_next_prize
            else:
                reinforced_p = p
            reinforced_prizes.append(reinforced_p)

        states = np.array(states)
        reinforced_prizes = np.array(reinforced_prizes)
        acting_model.train_on_batch(states, reinforced_prizes)



if LOAD:
    model_trainer = load_trainer(LOAD_FROM, NAME, action, training)
else:
    memory = SimpleMemory(MEMORY_SIZE)
    model_trainer = DQNTrainer(model, memory, action, training)

board = cb.ChessBoard()
for i in range(START_AT_STEP, TRAINING_STEPS):
    print("Step {} of {}".format(i+1, TRAINING_STEPS))
    model_trainer.take_action(board, get_epsilon(i))
    model_trainer.train(batch_size=BATCH, gamma=GAMMA, theta=THETA)
    if i % 1000 == 0:
        model_trainer.save("tmp", "{}_{}".format(NAME, i))

model_trainer.save("final", "{}_{}k".format(NAME, int(TRAINING_STEPS / 1000)))

