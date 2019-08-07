import chess
import keras.models
from chess_environment.chessboard import ChessBoard, IllegalMoveException
from engine import DQNChessEngine

board = ChessBoard()
model = keras.models.load_model("./model.h5")
ai_engine = DQNChessEngine(model)
while not board.game_over():
    if(board._current_state.turn):
        print(board._current_state)
        made_move = False
        while not made_move:
            from_square = ''
            while from_square == '':
                from_square = input("From: ").lower()
            to_square = ''
            while to_square == '':
                to_square = input("To: ").lower()
            from_square = chess.SQUARE_NAMES.index(from_square)
            to_square = chess.SQUARE_NAMES.index(to_square)
            try:
                move = chess.Move(from_square, to_square)
                board.make_move(move)
                made_move = True
            except IllegalMoveException:
                print("Illegal move!")
                made_move = False
    else:
        move, _ = ai_engine.choose_move(board, True)
        assert isinstance(move, chess.Move)
        board.make_move(move, True)
