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

    'legal_moves': 4,
    'player_to_move': 5,  # 1s if Black to move, 0s if White

    # Threat-based features
    'player_fours': 6,       # Moves that create 5-in-a-row (win)
    'opponent_fours': 7,     # Moves that block an opponent's 5-in-a-row
    'player_threes': 8,      # Moves that create an "open three" (_XXX_)
    'opponent_threes': 9,    # Moves that block an opponent's "open three"
    'player_threats_in_two': 10, # Moves that create >= 2 threats (fours or threes)
    'opponent_threats_in_two': 11, # Moves that block >= 2 opponent threats
}

class GomokuEncoder(Encoder):
    def __init__(self, board_size: Tuple[int, int]=(12, 12)):
        board_size = tuple(board_size)
        self.board_height, self.board_width = board_size
        self.num_planes = len(FEATURE_OFFSETS.keys())

        self._directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # H, V, Diag\, Diag/

    def name(self):
        return 'Gomoku'

    @staticmethod
    def _is_in_bounds(r, c, board_height, board_width):
        return 0 <= r < board_height and 0 <= c < board_width

    def _find_threats(self, r: int, c: int,
                        player_plane: torch.Tensor,
                        opponent_plane: torch.Tensor) -> Tuple[int, int]:
        num_fours = 0
        num_threes = 0

        for dr, dc in self._directions:
            line_str = ""
            for i in range(-4, 5): # 9-cell line centered on the move
                cr, cc = r + i * dr, c + i * dc

                if not self._is_in_bounds(cr, cc, self.board_height, self.board_width):
                    line_str += "B" # Border
                elif i == 0:
                    line_str += "P" # The hypothetical move
                elif player_plane[cr, cc] == 1:
                    line_str += "P" # Player's stone
                elif opponent_plane[cr, cc] == 1:
                    line_str += "O" # Opponent's stone
                else:
                    line_str += "E" # Empty

            if "PPPPP" in line_str:
                num_fours += 1
            if "EPPPPE" in line_str:
                num_threes += 1

        return num_fours, num_threes

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

        # Threat indices
        b_idx_p4, r_idx_p4, c_idx_p4 = [], [], []
        b_idx_o4, r_idx_o4, c_idx_o4 = [], [], []
        b_idx_p3, r_idx_p3, c_idx_p3 = [], [], []
        b_idx_o3, r_idx_o3, c_idx_o3 = [], [], []
        b_idx_p2t, r_idx_p2t, c_idx_p2t = [], [], []
        b_idx_o2t, r_idx_o2t, c_idx_o2t = [], [], []

        for i, game_state in enumerate(game_states):
            # Player to move
            if game_state.next_player == Player.black:
                board_tensors[i, FEATURE_OFFSETS['player_to_move']] = 1

            # Legal Moves
            legal_moves_list = game_state.legal_moves()
            for m in legal_moves_list:
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

            player = game_state.next_player
            opponent = Player.white if player == Player.black else Player.black

            pieces_planes = self.get_pieces_planes(game_state)
            player_plane = pieces_planes[BLACK_IDX] if player == Player.black else pieces_planes[WHITE_IDX]
            opponent_plane = pieces_planes[WHITE_IDX] if player == Player.black else pieces_planes[BLACK_IDX]

            for m in legal_moves_list:
                r, c = m.point.row - 1, m.point.col - 1

                # Check threats for player
                p_fours, p_threes = self._find_threats(r, c, player_plane, opponent_plane)

                if p_fours > 0:
                    b_idx_p4.append(i); r_idx_p4.append(r); c_idx_p4.append(c)
                if p_threes > 0:
                    b_idx_p3.append(i); r_idx_p3.append(r); c_idx_p3.append(c)
                if p_fours + p_threes >= 2:
                    b_idx_p2t.append(i); r_idx_p2t.append(r); c_idx_p2t.append(c)

                # Check threats for opponent
                o_fours, o_threes = self._find_threats(r, c, opponent_plane, player_plane)

                if o_fours > 0:
                    b_idx_o4.append(i); r_idx_o4.append(r); c_idx_o4.append(c)
                if o_threes > 0:
                    b_idx_o3.append(i); r_idx_o3.append(r); c_idx_o3.append(c)
                if o_fours + o_threes >= 2:
                    b_idx_o2t.append(i); r_idx_o2t.append(r); c_idx_o2t.append(c)

        # Assign current pieces
        board_tensors[b_idx_black_t0, FEATURE_OFFSETS['black_pieces_t0'], r_idx_black_t0, c_idx_black_t0] = 1
        board_tensors[b_idx_white_t0, FEATURE_OFFSETS['white_pieces_t0'], r_idx_white_t0, c_idx_white_t0] = 1

        # Assign previous pieces
        board_tensors[b_idx_black_t1, FEATURE_OFFSETS['black_pieces_t1'], r_idx_black_t1, c_idx_black_t1] = 1
        board_tensors[b_idx_white_t1, FEATURE_OFFSETS['white_pieces_t1'], r_idx_white_t1, c_idx_white_t1] = 1

        # Assign legal moves
        board_tensors[b_idx_legal, FEATURE_OFFSETS['legal_moves'], r_idx_legal, c_idx_legal] = 1

        # Assign threat planes
        board_tensors[b_idx_p4, FEATURE_OFFSETS['player_fours'], r_idx_p4, c_idx_p4] = 1
        board_tensors[b_idx_o4, FEATURE_OFFSETS['opponent_fours'], r_idx_o4, c_idx_o4] = 1
        board_tensors[b_idx_p3, FEATURE_OFFSETS['player_threes'], r_idx_p3, c_idx_p3] = 1
        board_tensors[b_idx_o3, FEATURE_OFFSETS['opponent_threes'], r_idx_o3, c_idx_o3] = 1
        board_tensors[b_idx_p2t, FEATURE_OFFSETS['player_threats_in_two'], r_idx_p2t, c_idx_p2t] = 1
        board_tensors[b_idx_o2t, FEATURE_OFFSETS['opponent_threats_in_two'], r_idx_o2t, c_idx_o2t] = 1

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
