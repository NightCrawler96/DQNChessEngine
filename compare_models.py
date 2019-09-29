import copy
import chess
import itertools
from keras.engine.saving import load_model
from chess_environment.chessboard import ChessBoard, Rewards
from engine import DQNChessEngine


class Table:
    Wins: int = 0
    Looses: int = 0
    Stales: int = 0
    Timeouts: int = 0

    def switch(self):
        buf = self.Looses
        self.Looses = self.Wins
        self.Wins = buf

    def __str__(self):
        return "Won: {} Lost: {} Stalemates: {} Timeouts: {}".format(self.Wins, self.Looses, self.Stales, self.Timeouts)


def make_move(player: DQNChessEngine, board: ChessBoard, black: bool, state_shape: tuple):
    move, _ = player.choose_move(board, black, state_shape=state_shape)
    assert isinstance(move, chess.Move)
    board.make_move(move, black)
    return board, board.get_result()


def get_state_shape(competitor: str):
    shapes = {
        "SimpleDQNv2 200k": (1, 1, 384)
    }
    if competitor in shapes.keys():
        return shapes[competitor]
    else:
        return 1, 384


def check_results(white, black, white_table, result):
    if result == '1-0':
        white_table.Wins += 1
        print(white, "Won in turn", turns)
    elif result == '0-1':
        white_table.Looses += 1
        print(black, "Won in turn", turns)
    elif result == '1/2-1/2':
        white_table.Stales += 1
        print("Stalemate")
    elif result == 'Timeout':
        white_table.Timeouts += 1
    else:
        return False, white_table

    return True, white_table


competitors_paths = {
    "SimpleDQNv2 200k": "models/SimpleDQNv2/SimpleDQNv2_200k.h5",
    "LeakyDQNv0 120k target ": "models/LeakyDQNv0/FinalModel/LeakyDQNv0_120k_target.h5f",
    "BuzdyganDQNv0 210k target": "models/BuzdyganDQNv0/FinalModel/BuzdyganDQNv0_210k_active.h5f",
    "BuzdyganDQNv1 150k target": "models/BuzdyganDQNv1/FinalModel/BuzdyganDQNv1_150k_target.h5f",
    "BorsukDQNv0 103k target": "models/BorsukDQNv0/FinalModel/BorsukDQNv0_103000_target.h5f",
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
            finished, white_table = check_results(white_k, black_k, white_table, result)
            if finished:
                break
            board, result = make_move(black, board, black=True, state_shape=black_state_shape)
            finished, white_table = check_results(white_k, black_k, white_table, result)
            if finished:
                break

    results[white_k].append((black_k, copy.deepcopy(white_table)))
    white_table.switch()
    results[black_k].append((white_k, white_table))
for r_k in results.keys():
    print(r_k, "results:")
    list_of_results = results[r_k]
    for r in list_of_results:
        print(r[0], str(r[1]))
