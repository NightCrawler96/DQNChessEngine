import numpy as np
import chess
import scipy.sparse
import re


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

    def _fen_to_numbers(self, board_fen):
        assert isinstance(board_fen, str)
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
    def _encode_field(number_field):
        assert isinstance(number_field, int)
        coded_field = np.zeros(6).tolist()
        if number_field != 0:
            sign = 1 if number_field > 0 else -1
            coded_field[abs(number_field) - 1] = sign
        return coded_field


