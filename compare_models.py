import copy
import chess
import itertools
from keras.engine.saving import load_model
from chess_environment.chessboard import ChessBoard, STALEMATE, CHECKMATE
from engine import DQNChessEngine


class Table:
    Wins: int = 0
    Looses: int = 0
    Stales: int = 0

    def switch(self):
        buf = self.Looses
        self.Looses = self.Wins
        self.Wins = buf

    def __str__(self):
        return "Won: {} Lost: {} Draws: {}".format(self.Wins, self.Looses, self.Stales)


def make_move(player: DQNChessEngine, board: ChessBoard, black: bool, state_shape: tuple):
    move, _ = player.choose_move(board, black, state_shape=state_shape)
    assert isinstance(move, chess.Move)
    board.make_move(move, black)
    return board, board.get_results()


def get_state_shape(competitor: str):
    shapes = {
        "SimpleDQNv2 200k": (1, 1, 384)
    }
    if competitor in shapes.keys():
        return shapes[competitor]
    else:
        return 1, 384


competitors_paths = {
    # "SimpleDQNv2 200k": "models/SimpleDQNv2/SimpleDQNv2_200k.h5",
    "LeakyDQNv0 120k": "models/LeakyDQNv0/FinalModel/LeakyDQNv0_120k_target.h5f",
    # "LeakyDQNv0 20k": "models/LeakyDQNv0/IntermediateModels/LeakyDQNv0_20000_target.h5f",
    # "LeakyDQNv0 60k": "models/LeakyDQNv0/IntermediateModels/LeakyDQNv0_60000_target.h5f",
    "LeakyDQNv0 initial": "models/LeakyDQNv0/IntermediateModels/LeakyDQNv0_0_target.h5f",
}
competitors = {}
for competitor, path in zip(competitors_paths.keys(), competitors_paths.values()):
    model = load_model(path)
    competitors[competitor] = DQNChessEngine(model)


GAMES_PER_PAIR = 2

pairs = list(itertools.combinations(competitors.keys(), 2))
results = {}
for k in competitors.keys():
    results[k] = []

for p in pairs:
    white_table = Table()
    white_k, black_k = "", ""
    for game in range(GAMES_PER_PAIR):
        white_k = p[0] if game % 2 == 0 else p[1]
        black_k = p[1] if game % 2 == 0 else p[0]
        print(white_k, black_k, game)
        white_state_shape = (get_state_shape(white_k))
        black_state_shape = (get_state_shape(black_k))
        white = competitors[white_k]
        black = competitors[black_k]
        board = ChessBoard()
        turns = 0
        while True:
            board, result = make_move(white, board, black=False, state_shape=white_state_shape)
            if result == CHECKMATE:
                white_table.Wins += 1
                print(white_k, "Won in turn", turns)
                break
            if result == STALEMATE :
                white_table.Stales += 1
                print("Draw")
                break
            board, result = make_move(black, board, black=True, state_shape=black_state_shape)
            if result == CHECKMATE:
                white_table.Looses += 1
                print(black_k, "Won in turn", turns)
                break
            turns += 1
            if result == STALEMATE or turns > 200:
                white_table.Stales += 1
                print("Draw")
                break
    results[white_k].append((black_k, copy.deepcopy(white_table)))
    white_table.switch()
    results[black_k].append((white_k, white_table))
for r_k in results.keys():
    print(r_k, "results:")
    list_of_results = results[r_k]
    for r in list_of_results:
        print(r[0], str(r[1]))
