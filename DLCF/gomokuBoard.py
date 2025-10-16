import copy

from typing import Dict, List

from DLCF import zobrist
from DLCF.DLCFtypes import Player, Point, Move, GameStateTemplate

# Number of stones you should place in a row to win
LENGTH_TO_WIN = 5


class Board():
    def __init__(self, num_rows:int, num_cols:int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid:Dict[Point, Player] = {}
        self._hash = zobrist.EMPTY_BOARD

    def zobrist_hash(self):
        return self._hash

    def place_stone(self, player:Player, point:Point):
        assert 0 <= point.col < self.num_cols+1, "Column out of bounds"
        assert 0 <= point.row < self.num_rows+1, "Row out of bounds"
        assert self._grid.get(point) is None, "Point is not empty"

        self._grid[point] = player
        self._hash ^= zobrist.HASH_CODE[point, player]

    def is_on_grid(self, point:Point) -> bool:
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def is_column_full(self, col: int):
        for r in range(1, self.num_cols+1):
            p = Point(r, col)
            if self._grid.get(p) is None:
                return False

        return True

    def get(self, point:Point):
        return self._grid.get(point)

    def visualize(self):
        for r in range(1, self.num_rows + 1):
            row_output = []
            for c in range(1, self.num_cols + 1):
                point = Point(row=r, col=c)
                player = self._grid.get(point)

                if player == Player.black:
                    row_output.append('O')
                elif player == Player.white:
                    row_output.append('X')
                else:
                    row_output.append('.')
            print(' '.join(row_output))

        print(' '.join(str(c) for c in range(1, self.num_cols + 1)))


class GameState(GameStateTemplate):
    def __init__(self, board: Board, next_player: Player, previous: "GameState", move: Move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous

        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())})

        self.last_move = move

    def apply_move(self, move:Move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    def is_over(self):
        if self.last_move is None:
            return False

        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False

        winner = self.compute_game_result()
        if winner is not None:
            return True

        if not self.legal_moves():
            return True

        return False

    def is_move_inside_grid(self, move: Move):
        return 1 <= move.point.row <= self.board.num_rows and \
            1 <= move.point.col <= self.board.num_cols

    def is_valid_move(self, move):
        return (
            self.board.get(move.point) is None and  # Grid point is empty
            self.is_move_inside_grid(move)  # Grid point is inside grid
        )

    def legal_moves(self):
        moves: List[Move] = []
        for col in range(1, self.board.num_cols+1):
            for row in range(1, self.board.num_rows+1):
                point = Point(row=row, col=col)
                if self.board._grid.get(point) is None:
                    move = Move.play(point)
                    if self.is_valid_move(move):
                        moves.append(move)

        return moves

    def winner(self):
        if not self.is_over():
            return None

        if self.last_move.is_resign:
            return self.next_player

        winner = self.compute_game_result(self)

        return winner

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            raise ValueError("Board size should be a tuple (h, w)")
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    @property
    def situation(self):
        return (self.next_player, self.board)

    def compute_game_result(self) -> Player:
        board = self.board

        for r in range(1, board.num_rows+1):
            for c in range(1, board.num_cols+1):
                start_point = Point(row=r, col=c)
                player = board.get(start_point)

                # If there's a piece here, check for a winning line
                if player is not None:
                    # Directions to check: right, down, down-right, down-left
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

                    for dr, dc in directions:
                        # Check if a line of <LENGTH_TO_WIN> exists starting from this point
                        is_a_win = True
                        for i in range(1, LENGTH_TO_WIN):
                            next_point = Point(row=r + i * dr, col=c + i * dc)
                            if board.get(next_point) != player:
                                is_a_win = False
                                break
                        if is_a_win:
                            return player
        # Draw
        if len(board._grid) == board.num_rows * board.num_cols:
            return None

        return None
