import torch

from typing import Tuple


from DLCF.encoders.base import Encoder
from DLCF.goboard import Move, GameState
from DLCF.DLCFtypes import Player, Point

BLACK_IDX = 0
WHITE_IDX = 1

FEATRE_OFFSETS = {
    'black_pieces': 0,
    'white_pieces': 1,
    'black_winning_moves': 2,
    'white_winning_moves': 3,
    'black_to_move': 4,
    'white_to_move': 5,
}

class ConnectFourEncoder(Encoder):
    def __init__(self, board_size: Tuple[int, int]=(6, 7)):
        board_size = tuple(board_size)
        self.board_height, self.board_width = board_size
        self.num_planes = len(FEATRE_OFFSETS.keys())

    def name(self):
        return 'connectFour'

    def encode(self, game_state: GameState):
        board_tensor = torch.zeros(self.shape())

        pieces_planes = self.get_pieces_planes(game_state)
        winning_moves_planes = self.get_winning_moves_planes(game_state)

        board_tensor[0:2] = pieces_planes
        board_tensor[2:4] = winning_moves_planes
        if game_state.next_player == Player.black:
            board_tensor[FEATRE_OFFSETS['black_to_move']] = 1
        else:
            board_tensor[FEATRE_OFFSETS['white_to_move']] = 1

        return board_tensor

    def get_pieces_planes(self, game_state: GameState):
        pieces_planes = torch.zeros(2, self.board_height, self.board_width)
        for row in range(1, self.board_height + 1):
            for col in range(1, self.board_width + 1):
                p = Point(row, col)
                player_at_pos = game_state.board.get(p)
                if player_at_pos is None:
                    continue
                elif player_at_pos == Player.white:
                    pieces_planes[WHITE_IDX][row-1][col-1] = 1
                elif player_at_pos == Player.black:
                    pieces_planes[BLACK_IDX][row-1][col-1] = 1

        return pieces_planes

    def get_winning_moves_planes(self, game_state: GameState):
        winning_moves_planes = torch.zeros(2, self.board_height, self.board_width)
        for m in game_state.legal_moves():
            winner = game_state.apply_move(m).winner
            if winner is None:
                continue
            # Winning move for white
            elif winner == Player.white:
                winning_moves_planes[WHITE_IDX][m.point.row-1][m.point.col-1] = 1
            # Winning move for black
            elif winner == Player.black:
                winning_moves_planes[BLACK_IDX][m.point.row-1][m.point.col-1] = 1

        return winning_moves_planes

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
