import chess
import chess.svg
import keras
from chess_environment.chessboard import ChessBoard
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QWidget, QApplication


# based on: https://stackoverflow.com/a/47329268/6708094
from engine import DQNChessEngine


class Window(QWidget):
    def __init__(self):
        super().__init__()

        model = keras.models.load_model("./models/SimpleDQNv2_200k.h5", compile=False)
        self.ai_engine = DQNChessEngine(model)
        self.board = chess.Board()
        self.chosen_piece = [None, None]

        self.board_size = 600
        self.coordinates = True
        self.margin = .05 * self.board_size if self.coordinates else 0
        self.sqr_size = (self.board_size - 2 * self.margin) / 8.

        self.setWindowTitle("DQNChess GUI")
        self.setGeometry(600, 200, self.board_size + self.margin, self.board_size + self.margin)

        self.board_svg = None
        self.svg_widget = QSvgWidget(parent=self)
        self.svgX = 10
        self.svgY = 10
        self.svg_widget.setGeometry(self.svgX, self.svgY, self.board_size, self.board_size)


    @pyqtSlot(QWidget)
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.svgX < event.x() <= self.svgX + self.board_size and \
                self.svgY < event.y() <= self.svgY + self.board_size:
            if event.buttons() == Qt.LeftButton:
                if self.svgX + self.margin < event.x() < self.svgX + self.board_size - self.margin and \
                        self.svgY + self.margin < event.y() < self.svgY + self.board_size - self.margin:
                    file = int((event.x() - (self.svgX + self.margin)) / self.sqr_size)
                    rank = 7 - int((event.y() - (self.svgY + self.margin)) / self.sqr_size)
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    coordinates = '{}{}'.format(chr(file + 97), str(rank +1))
                    if self.chosen_piece[0] is not None:
                        move = chess.Move.from_uci('{}{}'.format(self.chosen_piece[1], coordinates))
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            piece = None
                            coordinates = None
                            if self.board.is_game_over():
                                print("Player won")

                            cb_board = ChessBoard(self.board.fen())
                            ai_move, _ = self.ai_engine.choose_move(cb_board, flip=True)
                            flipped_board = self.board.mirror()
                            flipped_board.push(ai_move)
                            self.board = flipped_board.mirror()
                            if self.board.is_game_over():
                                print("AI won")

                    self.chosen_piece = [piece, coordinates]
            self.update()
        else:
            QWidget.mousePressEvent(self, event)

    @pyqtSlot(QWidget)
    def paintEvent(self, a0: QtGui.QPaintEvent):
        self.board_svg = chess.svg.board(self.board, size=self.board_size, coordinates=self.coordinates).encode("UTF-8")
        self.svg_widget.load(self.board_svg)


if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    app.exec()
