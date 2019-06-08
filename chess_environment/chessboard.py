import numpy as np
import chess
import re


class IllegalMoveException(Exception):
    def __init__(self):
        pass


ATTACK = 5
CHECKMATE = 100
STALEMATE = 75
IGNORE_GO = 420


class ChessBoard:
    def __init__(self, starting_fen=chess.STARTING_BOARD_FEN):
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

    def _encode_board(self, board: chess.Board):
        board = self._fen_to_numbers(board.fen())
        encoded_board = []
        for field in board:
            encoded_field = self._encode_field(field)
            encoded_board += encoded_field

        return encoded_board

    def get_moves(self, flip=False):
        board = self._current_state if not flip else self._current_state.mirror()
        possible_moves = board.legal_moves
        possible_states = []

        for move in possible_moves:
            future_board = board.copy()
            future_board.push(move)
            future_board = self._encode_board(future_board)
            possible_states.append(future_board)

        return possible_moves, possible_states

    def _check_attack(self, board: chess.Board, move: chess.Move):
        possible_attacks = board.attacks(move.to_square)
        if move.from_square in possible_attacks:
            self._attacked = True

    def make_move(self, move: chess.Move, flipped=False):
        if flipped:
            board = self._current_state.mirror()
            assert isinstance(board, chess.Board)
            if not board.is_legal(move):
                raise IllegalMoveException()
            self._check_attack(board, move)
            board.push(move)
            self._current_state = board.mirror()
        else:
            if not self._current_state.is_legal(move):
                raise IllegalMoveException()
            self._check_attack(self._current_state, move)
            self._current_state.push(move)

    """
    True = WHITE
    False = BLACK
    """
    def current_turn(self):
        return self._current_state.turn

    def get_results(self):
        if self._current_state.is_game_over():
            if self._current_state.is_checkmate():
                return CHECKMATE
            if self._current_state.is_stalemate():
                return STALEMATE
            if self._current_state.is_seventyfive_moves() or self._current_state.is_fivefold_repetition():
                return IGNORE_GO
            self._current_state = chess.Board(chess.STARTING_BOARD_FEN)
        if self._attacked:
            self._attacked = False
            return ATTACK
        return 0
