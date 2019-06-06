import chess

from chess_environment.chessboard import ChessBoard, IllegalMoveException

board = ChessBoard()

while not board._current_state.is_game_over():
    print(board._current_state)
    if board._current_state.turn:
        print("White (big)")
    else:
        print("Black (lower)")
    made_move = False
    while not made_move:
        from_square = input("From: ").lower()
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
