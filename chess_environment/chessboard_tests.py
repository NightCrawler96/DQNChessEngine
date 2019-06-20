import unittest
import chess
import chess_environment.chessboard as cb


class ChessBoardTests(unittest.TestCase):
    def setUp(self):
        self.board = cb.ChessBoard()

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

        actual_output = self.board._encode_board(board.fen())
        self.assertListEqual(actual_output, desired_output)

    def test_getting_possible_moves(self):
        inner_board = self.board._current_state
        moves, states, _ = self.board.get_moves()
        test_move = ""
        for m in moves:  # getting first element of dynamic list
            test_move = m
            break

        test_state = states[0]
        inner_board.push(test_move)
        inner_board_coded = self.board._encode_board(inner_board.fen())
        self.assertListEqual(test_state, inner_board_coded)

    def test_getting_possible_moves_flipped_board(self):
        inner_board = self.board._current_state
        moves, states, _ = self.board.get_moves(flip=True)
        test_move = None
        for m in moves:  # getting first element of dynamic list
            test_move = m
            break

        test_state = states[0]
        inner_board_flipped = inner_board.mirror()
        inner_board_flipped.push(test_move)
        inner_board_flipped_coded = self.board._encode_board(inner_board_flipped.fen())
        self.assertListEqual(test_state, inner_board_flipped_coded)

    def test_making_move(self):
        test_board = self.board._current_state.copy()
        move = None
        for m in test_board.legal_moves:
            move = m
            break

        test_board.push(move)
        self.board.make_move(move)
        self.assertEqual(self.board._current_state.fen(), test_board.fen())

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
        with self.assertRaises(cb.IllegalMoveException):
            self.board.make_move(move)

    def test_attack_detection(self):
        fen_code = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        board = cb.ChessBoard(fen_code)
        move = chess.Move(chess.E4, chess.D5)
        board.make_move(move)
        self.assertTrue(board._attacked)

    def test_reward_attack(self):
        fen_code = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        board = cb.ChessBoard(fen_code)
        move = chess.Move(chess.E4, chess.D5)
        board.make_move(move)
        self.assertEqual(board.get_results(), cb.ATTACK)

    def test_checkmate_detection(self):
        fen_code = "8/8/8/5K1k/8/8/8/6R1 w k - 0 1"
        board = cb.ChessBoard(fen_code)
        move = chess.Move(chess.G1, chess.H1)
        board.make_move(move)
        self.assertEqual(board.get_results(), cb.CHECKMATE)


if __name__ == "__main__":
    unittest.main()
