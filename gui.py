import chess
import chess.svg
import keras
from chess_environment.chessboard import ChessBoard
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel

# based on: https://stackoverflow.com/a/47329268/6708094
from engine import DQNChessEngine


class Window(QWidget):
    def __init__(self):
        super().__init__()

        model = keras.models.load_model("final/BuzdyganDQNv0_210k_target.h5f", compile=False)
        self.ai_engine = DQNChessEngine(model)
        self.board: chess.Board = chess.Board()
        self.chosen_piece = [None, None]
        self.last_ai_move: chess.Move = None
        self.result = None
        self.svg_clickable: bool = True

        # Geometry
        self.board_size = 600
        self.coordinates = True
        self.margin = .05 * self.board_size if self.coordinates else 0
        self.sqr_size = (self.board_size - 2 * self.margin) / 8.

        self.button_width = 100
        self.button_height = 50

        self.window_width = self.board_size + self.margin + self.button_width + 20
        self.window_height = self.board_size + self.margin

        self.svgX = 10
        self.svgY = 10

        self.setWindowTitle("DQNChess GUI")
        self.setFixedSize(self.window_width, self.window_height)

        self.board_svg = None
        self.svg_widget = QSvgWidget(parent=self)
        self.svg_widget.setGeometry(self.svgX, self.svgY, self.board_size, self.board_size,)

        self.restart_button = QPushButton("Play next round", self)
        self.restart_button.setGeometry(
            self.board_size + self.margin + 5,
            self.margin,
            self.button_width,
            self.button_height)
        self.restart_button.clicked.connect(self.restart)

        self.text_label = QLabel("Start playing.", self)
        self.text_label.setGeometry(
            self.board_size + self.margin + 5,
            self.margin * 2 + self.button_height,
            100,
            20
        )

    def restart(self):
        self.board.reset()
        self.last_ai_move = None
        self.svg_clickable = True
        self.text_label.setText("Start playing.")

    def _is_game_over(self):
        if self.board.is_game_over():
            self.result = self.board.result()
            self.svg_clickable = False
            return True
        else:
            return False

    def _can_next_player_move(self):
        return not self._is_game_over()

    @staticmethod
    def _flip_move(move: chess.Move):
        uci = move.uci()
        from_column = uci[0]
        from_row = 9 - int(uci[1])
        to_column = uci[2]
        to_row = 9 - int(uci[3])
        flipped_move = chess.Move.from_uci("{}{}{}{}".format(from_column, from_row, to_column, to_row))
        return flipped_move

    @pyqtSlot(QWidget)
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.svgX < event.x() <= self.svgX + self.board_size and \
                self.svgY < event.y() <= self.svgY + self.board_size:
            if event.buttons() == Qt.LeftButton:
                if self.svgX + self.margin < event.x() < self.svgX + self.board_size - self.margin and \
                        self.svgY + self.margin < event.y() < self.svgY + self.board_size - self.margin and \
                        self.svg_clickable:
                    file = int((event.x() - (self.svgX + self.margin)) / self.sqr_size)
                    rank = 7 - int((event.y() - (self.svgY + self.margin)) / self.sqr_size)
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    coordinates = '{}{}'.format(chr(file + 97), str(rank + 1))
                    if self.chosen_piece[0] is not None:
                        move = chess.Move.from_uci('{}{}'.format(self.chosen_piece[1], coordinates))
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            piece = None
                            coordinates = None
                            can_ai_move = self._can_next_player_move()
                            if can_ai_move:
                                cb_board = ChessBoard(self.board.fen())
                                ai_move, _ = self.ai_engine.choose_move(cb_board, flip=True)
                                ai_move = self._flip_move(ai_move)
                                self.board.push(ai_move)
                                self.last_ai_move = ai_move
                                self._can_next_player_move()
                    self.chosen_piece = [piece, coordinates]
            self.update()
        else:
            QWidget.mousePressEvent(self, event)

    @pyqtSlot(QWidget)
    def paintEvent(self, a0: QtGui.QPaintEvent):
        king_attacked_by = None
        if self._is_game_over():
            if self.result == '1-0':
                king_square = self.board.king(False)
                king_attacked_by = self.board.attackers(True, king_square)
                self.text_label.setText("You have won!")
            elif self.result == '0-1':
                king_square = self.board.king(True)
                king_attacked_by = self.board.attackers(False, king_square)
                self.text_label.setText("You have lost! :(")
            else:
                self.text_label.setText("Stalemate")
        self.board_svg = chess.svg.board(
            self.board,
            size=self.board_size,
            coordinates=self.coordinates,
            lastmove=self.last_ai_move,
            squares=king_attacked_by
        ).encode("UTF-8")
        self.svg_widget.load(self.board_svg)


if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    app.exec()
