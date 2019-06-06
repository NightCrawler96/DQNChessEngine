import unittest

import chess

from chess_environment.chessboard import ChessBoard, IllegalMoveException


class ChessBoardTests(unittest.TestCase):
    def setUp(self):
        self.board = ChessBoard()

    def test_fen_to_numbers(self):
        fen_code = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        desired_output = \
            [4, 2, 3, 5, 6, 3, 2, 4] + \
            [1, 1, 1, 1, 1, 1, 1, 1] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [-1, -1, -1, -1, -1, -1, -1, -1] + \
            [-4, -2, -3, -5, -6, -3, -2, -4]

        actual_output = self.board._fen_to_numbers(fen_code)
        self.assertListEqual(actual_output, desired_output)

    def test_coding_fields(self):
        numbers = [0, 1, 3, -4, -6]
        codes = [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, -1]
        ]

        for n, c in zip(numbers, codes):
            actual_code = self.board._encode_field(n)
            self.assertListEqual(actual_code, c)

    def test_coding_board(self):
        board = chess.Board(chess.STARTING_BOARD_FEN)

        desired_numbers = \
            [4, 2, 3, 5, 6, 3, 2, 4] + \
            [1, 1, 1, 1, 1, 1, 1, 1] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [0, 0, 0, 0, 0, 0, 0, 0] + \
            [-1, -1, -1, -1, -1, -1, -1, -1] + \
            [-4, -2, -3, -5, -6, -3, -2, -4]

        desired_output = []
        for number in desired_numbers:
            desired_output += self.board._encode_field(number)

        actual_output = self.board._encode_board(board)
        self.assertListEqual(actual_output, desired_output)

    def test_getting_possible_moves(self):
        inner_board = self.board._current_state
        moves, states = self.board.get_moves()
        test_move = ""
        for m in moves:  # getting first element of dynamic list
            test_move = m
            break

        test_state = states[0]
        inner_board.push(test_move)
        inner_board_coded = self.board._encode_board(inner_board)
        self.assertListEqual(test_state, inner_board_coded)

    def test_getting_possible_moves_flipped_board(self):
        inner_board = self.board._current_state
        moves, states = self.board.get_moves(flip=True)
        test_move = None
        for m in moves:  # getting first element of dynamic list
            test_move = m
            break

        test_state = states[0]
        inner_board_flipped = inner_board.mirror()
        inner_board_flipped.push(test_move)
        inner_board_flipped_coded = self.board._encode_board(inner_board_flipped)
        self.assertListEqual(test_state, inner_board_flipped_coded)

    def test_making_move_flipped_board(self):
        test_board = self.board._current_state.copy()
        test_board = test_board.mirror()
        move = None
        for m in test_board.legal_moves:
            move = m
            break

        test_board.push(move)
        self.board.make_move(move, flipped=True)
        self.assertEqual(self.board._current_state.fen(), test_board.mirror().fen())

    def test_illegal_move(self):
        move = chess.Move(chess.A1, chess.A2)
        with self.assertRaises(IllegalMoveException):
            self.board.make_move(move)


if __name__ == "__main__":
    unittest.main()
