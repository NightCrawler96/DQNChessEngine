import keras
import numpy as np
from chess_environment.chessboard import ChessBoard


class DQNChessEngine:
    def __init__(self, model: keras.Model):
        self._model = model

    def choose_move(self, board: ChessBoard, flip=False, state_shape: tuple=(1, 384)):
        moves, states, _ = board.get_moves(flip=flip)
        highest_prize = 0
        best_move = None
        best_state = None
        for m, s in zip(moves, states):
            prize = self._model.predict(np.array(s).reshape(state_shape))
            if prize > highest_prize or best_move is None:
                highest_prize = prize
                best_move = m
                best_state = s
        return best_move, best_state
