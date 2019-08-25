import chess_environment.chessboard as cb
import keras
import numpy as np

from dqn_tools.memory import SimpleMemory
from keras.initializers import RandomNormal
from keras.layers import Dense, LeakyReLU
from keras.regularizers import l2
from models.model_template import ModelTemplate
from training_tools import DQNChessRecord


class BuzdyganDQNv0Templte(ModelTemplate):
    NAME = "BuzdyganDQNv0"
    START_AT_STEP = 150000
    TRAINING_STEPS = 210000
    MEMORY_SIZE = 50000
    START_TRAINING_AT = 2000
    BATCH = 32
    GAMMA = 0.99
    THETA = 0.05
    EPSILON = 0.2
    EPSILON_THRESHOLD = START_TRAINING_AT * 1.01
    SAVE_PER_STEPS = 1000

    @staticmethod
    def new_model(seed: int):
        weight_decay = l2(1e-2)
        weight_initializer = RandomNormal(mean=0., stddev=0.02, seed=seed)
        model = keras.Sequential([
            Dense(200, input_shape=(384,),
                  kernel_initializer=weight_initializer,
                  bias_initializer=weight_initializer,
                  kernel_regularizer=weight_decay,
                  bias_regularizer=weight_decay),
            LeakyReLU(alpha=0.01),
            Dense(300,
                  kernel_initializer=weight_initializer,
                  bias_initializer=weight_initializer,
                  kernel_regularizer=weight_decay,
                  bias_regularizer=weight_decay),
            LeakyReLU(alpha=0.01),
            Dense(1, activation="linear",
                  kernel_initializer=weight_initializer,
                  bias_initializer=weight_initializer,
                  kernel_regularizer=weight_decay,
                  bias_regularizer=weight_decay)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def get_epsilon(self, step: int):
        return 1 if step < self.EPSILON_THRESHOLD else self.EPSILON

    @staticmethod
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

        return best_move, best_state, best_state_fen, highest_prize

    def action(self, acting_model: keras.Model, models_memory: SimpleMemory, environment: cb.ChessBoard, epsilon):
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
            best_move, best_state, best_state_fen, _ = self.choose_action(acting_model, moves, states, fens)
        # make move
        environment.make_move(best_move, flip)
        real_prize = environment.get_results()
        best_state = np.array(best_state).reshape((384,))
        real_prize = np.array([real_prize]).reshape((1, 1))
        if real_prize == cb.IGNORE_GO:
            return
        record = DQNChessRecord()
        record.state = best_state
        record.fen = best_state_fen
        record.reward = real_prize
        models_memory.add(record)

    def training(self,
            acting_model: keras.Model,
            target_model: keras.Model,
            models_memory: SimpleMemory,
            batch_size: int,
            gamma: float):
        training_batch = models_memory.get_batch(batch_size, min_rows=self.START_TRAINING_AT)
        if training_batch is not None:
            samples = [[record.state, record.reward, record.fen] for record in training_batch]
            states, prizes, fens = list(map(list, zip(*samples)))
            reinforced_prizes = []
            for p, f in zip(prizes, fens):
                training_board = cb.ChessBoard(starting_fen=f)
                p = p[0]
                if not training_board.game_over():
                    # predict opponent's move
                    opponents_next_moves, opponents_next_states, opponents_next_fens = \
                        training_board.get_moves(flip=True)
                    opponents_move, _, _, _ = self.choose_action(
                        target_model, opponents_next_moves, np.array(opponents_next_states), opponents_next_fens)
                    training_board.make_move(opponents_move, flipped=True)
                    opponents_prize = training_board.get_results()
                    if opponents_prize > cb.ATTACK:
                        reinforced_p = p - gamma * opponents_prize
                    else:
                        # get expected next move's reward
                        possible_moves, possible_states, possible_fens = training_board.get_moves()
                        _, _, _, estimated_next_prize = self.choose_action(
                            target_model, possible_moves, np.array(possible_states), possible_fens)
                        estimated_next_prize = \
                            estimated_next_prize if isinstance(estimated_next_prize, int) else estimated_next_prize[0]

                        reinforced_p = p + gamma * (estimated_next_prize - opponents_prize)
                else:
                    reinforced_p = p
                reinforced_prizes.append(reinforced_p)

            states = np.array(states)
            reinforced_prizes = np.array(reinforced_prizes)
            acting_model.train_on_batch(states, reinforced_prizes)
