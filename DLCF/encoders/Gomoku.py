import torch

from typing import Tuple, List

from DLCF.DLCFtypes import Player, Point
from DLCF.encoders.base import Encoder
from DLCF.getGameState import GameStateTemplate

BLACK_IDX = 0
WHITE_IDX = 1

# TODO: More features
FEATURE_OFFSETS = {
    'black_pieces': 0,
    'white_pieces': 1,
    'black_winning_moves': 2,
    'white_winning_moves': 3,
    'black_to_move': 4,
    'white_to_move': 5,
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

        # Piece indices
        b_idx_black, r_idx_black, c_idx_black = [], [], []
        b_idx_white, r_idx_white, c_idx_white = [], [], []

        # Winning move indices
        b_idx_win_black, r_idx_win_black, c_idx_win_black = [], [], []
        b_idx_win_white, r_idx_win_white, c_idx_win_white = [], [], []

        for i, game_state in enumerate(game_states):
            # --- A: Player to move ---
            if game_state.next_player == Player.black:
                board_tensors[i, FEATURE_OFFSETS['black_to_move']] = 1
            else:
                board_tensors[i, FEATURE_OFFSETS['white_to_move']] = 1

            # --- B: Piece positions ---
            for row in range(1, self.board_height + 1):
                for col in range(1, self.board_width + 1):
                    player = game_state.board.get(Point(row, col))
                    if player == Player.black:
                        b_idx_black.append(i)
                        r_idx_black.append(row - 1)
                        c_idx_black.append(col - 1)
                    elif player == Player.white:
                        b_idx_white.append(i)
                        r_idx_white.append(row - 1)
                        c_idx_white.append(col - 1)

            # --- C: Winning moves ---
            for m in game_state.legal_moves():
                winner = game_state.apply_move(m).winner
                if winner == Player.black:
                    b_idx_win_black.append(i)
                    r_idx_win_black.append(m.point.row - 1)
                    c_idx_win_black.append(m.point.col - 1)
                elif winner == Player.white:
                    b_idx_win_white.append(i)
                    r_idx_win_white.append(m.point.row - 1)
                    c_idx_win_white.append(m.point.col - 1)

        # Assign pieces
        board_tensors[b_idx_black, FEATURE_OFFSETS['black_pieces'], r_idx_black, c_idx_black] = 1
        board_tensors[b_idx_white, FEATURE_OFFSETS['white_pieces'], r_idx_white, c_idx_white] = 1

        # Assign winning moves
        board_tensors[b_idx_win_black, FEATURE_OFFSETS['black_winning_moves'], r_idx_win_black, c_idx_win_black] = 1
        board_tensors[b_idx_win_white, FEATURE_OFFSETS['white_winning_moves'], r_idx_win_white, c_idx_win_white] = 1

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

    def get_winning_moves_planes(self, game_state: GameStateTemplate):
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

def create(board_size=(12,12)):
    return GomokuEncoder(board_size)
