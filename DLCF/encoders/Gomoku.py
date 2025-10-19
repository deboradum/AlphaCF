import torch
from typing import Tuple, List

from DLCF.DLCFtypes import Player, Point
from DLCF.encoders.base import Encoder
from DLCF.getGameState import GameStateTemplate

BLACK_IDX = 0
WHITE_IDX = 1

FEATURE_OFFSETS = {
    # Current board state
    'black_pieces_t0': 0,
    'white_pieces_t0': 1,

    # Previous board state
    'black_pieces_t1': 2,
    'white_pieces_t1': 3,

    # Other features
    'legal_moves': 4,
    'black_to_move': 5,
    'white_to_move': 6,
}

class GomokuEncoder(Encoder):
    def __init__(self, board_size: Tuple[int, int]=(12, 12)):
        board_size = tuple(board_size)
        self.board_height, self.board_width = board_size
        self.num_planes = len(FEATURE_OFFSETS.keys())

    def name(self):
        return 'Gomoku'

    def encode(self, game_states: List[GameStateTemplate]):
        batch_size = len(game_states)

        board_tensors = torch.zeros(
            (batch_size, self.num_planes, self.board_height, self.board_width),
            dtype=torch.float32
        )

        # Piece indices (t=0)
        b_idx_black_t0, r_idx_black_t0, c_idx_black_t0 = [], [], []
        b_idx_white_t0, r_idx_white_t0, c_idx_white_t0 = [], [], []

        # Piece indices (t-1)
        b_idx_black_t1, r_idx_black_t1, c_idx_black_t1 = [], [], []
        b_idx_white_t1, r_idx_white_t1, c_idx_white_t1 = [], [], []

        # Legal move indices
        b_idx_legal, r_idx_legal, c_idx_legal = [], [], []

        for i, game_state in enumerate(game_states):
            # Player to move
            if game_state.next_player == Player.black:
                board_tensors[i, FEATURE_OFFSETS['black_to_move']] = 1
            else:
                board_tensors[i, FEATURE_OFFSETS['white_to_move']] = 1

            # Legal Moves
            for m in game_state.legal_moves():
                b_idx_legal.append(i)
                r_idx_legal.append(m.point.row - 1)
                c_idx_legal.append(m.point.col - 1)

            # Current piece positions
            for row in range(1, self.board_height + 1):
                for col in range(1, self.board_width + 1):
                    player = game_state.board.get(Point(row, col))
                    if player == Player.black:
                        b_idx_black_t0.append(i)
                        r_idx_black_t0.append(row - 1)
                        c_idx_black_t0.append(col - 1)
                    elif player == Player.white:
                        b_idx_white_t0.append(i)
                        r_idx_white_t0.append(row - 1)
                        c_idx_white_t0.append(col - 1)

            # Previous piece positions
            previous_state = getattr(game_state, 'previous_state', None)
            if previous_state:
                for row in range(1, self.board_height + 1):
                    for col in range(1, self.board_width + 1):
                        player = previous_state.board.get(Point(row, col))
                        if player == Player.black:
                            b_idx_black_t1.append(i)
                            r_idx_black_t1.append(row - 1)
                            c_idx_black_t1.append(col - 1)
                        elif player == Player.white:
                            b_idx_white_t1.append(i)
                            r_idx_white_t1.append(row - 1)
                            c_idx_white_t1.append(col - 1)

        # Assign current pieces
        board_tensors[b_idx_black_t0, FEATURE_OFFSETS['black_pieces_t0'], r_idx_black_t0, c_idx_black_t0] = 1
        board_tensors[b_idx_white_t0, FEATURE_OFFSETS['white_pieces_t0'], r_idx_white_t0, c_idx_white_t0] = 1

        # Assign previous pieces
        board_tensors[b_idx_black_t1, FEATURE_OFFSETS['black_pieces_t1'], r_idx_black_t1, c_idx_black_t1] = 1
        board_tensors[b_idx_white_t1, FEATURE_OFFSETS['white_pieces_t1'], r_idx_white_t1, c_idx_white_t1] = 1

        # legal moves
        board_tensors[b_idx_legal, FEATURE_OFFSETS['legal_moves'], r_idx_legal, c_idx_legal] = 1

        return board_tensors

    def get_pieces_planes(self, game_state: GameStateTemplate):
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

def create(board_size=(12,12)):
    return GomokuEncoder(board_size)
