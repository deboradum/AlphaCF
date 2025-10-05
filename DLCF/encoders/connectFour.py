import torch

from typing import Tuple


from DLCF.encoders.base import Encoder
from DLCF.goboard import Move, GameState
from DLCF.types import Player, Point

FEATRE_OFFSETS = {
    'stone_color': 0,
    'ones': 3,
    'zeros': 4,
    'sensibleness': 5,
    'turns_since': 6,
    'liberties': 14,
    'liberties_after': 22,
    'capture_size': 30,
    'self_atari_size': 38,
    'ladder_capture': 46,
    'ladder_escape': 47,
    'current_player_color': 48
}

class ConnectFourEncoder(Encoder):
    def __init__(self, board_size: Tuple[int, int]=(6, 7)):
        self.board_width, self.board_height = board_size
        self.num_planes = len(FEATRE_OFFSETS.keys())

    def name(self):
        return 'ConnectFour'

    def encode(self, game_state: GameState):
        board_tensor = torch.zeros(self.shape())

        if game_state.next_player == Player.black:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1

        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                    Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1
        return board_tensor

    def encode_point(self, point: Point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index: int):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

    def ones(self):
        return torch.ones((1, self.board_height, self.board_width))

    def zeros(self):
        return torch.zeros((1, self.board_height, self.board_width))

def create(board_size=(6,7)):
    return ConnectFourEncoder(board_size)
