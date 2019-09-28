from enum import Enum

import numpy as np
import chess
import re


class IllegalMoveException(Exception):
    def __init__(self):
        pass


class Rewards(Enum):
    ATTACK = 15
    CHECKMATE = 100
    STALEMATE = -10
    TURN_PENALTY = -.5


class ChessBoard:
    def __init__(self, starting_fen=chess.STARTING_BOARD_FEN, start_at: int = 0, max_turns: int = 50):
        self._current_state = chess.Board(starting_fen)
        self._dictionary = dict({
            'p': chess.PAWN,
            'n': chess.KNIGHT,
            'r': chess.ROOK,
            'b': chess.BISHOP,
            'q': chess.QUEEN,
            'k': chess.KING
        })
        self._attacked = False
        self._turn_num = start_at
        self._reset_at = max_turns

    def _fen_to_numbers(self, board_fen: str):
        fen_rows = re.split("\/|\ ", board_fen)[:8]
        board_numbers = []

        for row in fen_rows:
            for field in row:
                multiplier = 1 if field.islower() else -1
                field = field.lower()
                if field in self._dictionary.keys():
                    field_number = self._dictionary[field] * multiplier
                    board_numbers.append(field_number)
                else:
                    zero_fields_num = int(field)
                    zero_fields = np.zeros(zero_fields_num, dtype=int).tolist()
                    board_numbers += zero_fields

        return board_numbers

    @staticmethod
    def _encode_field(number_field: int):
        coded_field = np.zeros(6, dtype=int).tolist()
        if number_field != 0:
            sign = 1 if number_field > 0 else -1
            coded_field[abs(number_field) - 1] = sign
        return coded_field

    def _encode_board(self, board_fen: str):
        board = self._fen_to_numbers(board_fen)
        encoded_board = []
        for field in board:
            encoded_field = self._encode_field(field)
            encoded_board += encoded_field

        return encoded_board

    def get_moves(self, flip=False):
        board = self._current_state if not flip else self._current_state.mirror()
        possible_moves = board.legal_moves
        possible_states = []
        possible_states_fens = []

        for move in possible_moves:
            future_board = board.copy()
            future_board.push(move)
            future_board_fen = future_board.fen()
            future_board = self._encode_board(future_board_fen)
            possible_states.append(future_board)
            possible_states_fens.append(future_board_fen)

        return possible_moves, possible_states, possible_states_fens

    def _check_attack(self, board: chess.Board, move: chess.Move, color: bool):
        possible_attacks = board.attackers(not color, move.from_square)
        if move.to_square in possible_attacks:
            self._attacked = True

    def make_move(self, move: chess.Move, flipped=False, color=True):
        if flipped:
            board = self._current_state.mirror()
        else:
            board = self._current_state
        assert isinstance(board, chess.Board)
        if not board.is_legal(move):
            raise IllegalMoveException()
        self._check_attack(board, move, color)
        board.push(move)
        if flipped:
            self._current_state = board.mirror()

    """
    True = WHITE
    False = BLACK
    """
    def current_turn(self):
        return self._current_state.turn

    def timeout(self):
        return self._turn_num >= self._reset_at

    def get_reward(self, reset: bool = True):
        reward = Rewards.TURN_PENALTY.value * self._turn_num
        if self._current_state.is_game_over():
            if self._current_state.is_checkmate():
                reward += Rewards.CHECKMATE.value
            elif self._current_state.is_stalemate():
                reward += Rewards.STALEMATE.value
            if reset:
                self.reset()
        elif self.timeout() and reset:
            self.reset()

        if self._attacked:
            self._attacked = False
            reward += Rewards.ATTACK.value

        return reward

    def get_result(self):
        if self.game_over():
            return self._current_state.result()
        elif self.timeout():
            return "Timeout"
        else:
            return None

    def game_over(self):
        return self._current_state.is_game_over()

    def turn(self):
        return self._current_state.turn

    def get_piece(self, square):
        return self._current_state.piece_at(square)

    def reset(self):
        self._current_state = chess.Board(chess.STARTING_BOARD_FEN)
        self._turn_num = 0

    def get_king_attackers(self, color: bool):
        king_square = self._current_state.king(color)
        king_attacked_by = self._current_state.attackers(not color, king_square)
        return king_attacked_by

    def get_board(self):
        return self._current_state

    def turn_number(self):
        return self._turn_num
