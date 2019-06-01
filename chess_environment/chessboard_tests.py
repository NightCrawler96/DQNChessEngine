import unittest

from chess_environment.chessboard import ChessBoard


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

        actual_output = self.board. _fen_to_numbers(fen_code)
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


if __name__ == "__main__":
    unittest.main()
