import enum
from typing import NamedTuple, List


class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


class Point(NamedTuple):
    row: int
    col: int

    def neighbors(self) -> list["Point"]:
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]


class Move():
    def __init__(self, point:Point=None):
        assert (point is not None)
        self.point = point
        self.is_play = (self.point is not None)

    @classmethod
    def play(cls, point:Point):
        return Move(point=point)


class GameStateTemplate:
    def apply_move(self, move:Move):
        raise NotImplementedError()

    def is_over(self):
        raise NotImplementedError()

    def legal_moves(self) -> List[Move]:
        raise NotImplementedError()

    def winner(self):
        raise NotImplementedError()

    @classmethod
    def new_game(cls, board_size):
        raise NotImplementedError()

    @property
    def situation(self):
        raise NotImplementedError()

    def compute_game_result(self) -> Player:
        raise NotImplementedError()
