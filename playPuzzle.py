import json
import torch
import argparse

from DLCF import rl
from enum import Enum
from Model import Model
from typing import List
from dataclasses import dataclass
from DLCF.DLCFtypes import Player, Move, Point
from DLCF.connectFourBoard import Board, GameState
import numpy as np

np.set_printoptions(linewidth=200, precision=8, suppress=True)


class PuzzleDifficulty(str, Enum):
    SIMPLE = "Simple"
    TOUGH = "Tough"
    HARD = "Harder"
    SEVERE = "Severe"

@dataclass(frozen=True)
class PuzzleBoard:
    num_rows: int
    num_cols: int
    grid: list[list[int]]

@dataclass(frozen=True)
class PuzzleMove:
    player: int
    column: int
    row: int

@dataclass(frozen=True)
class Puzzle:
    puzzleId: int
    difficulty: PuzzleDifficulty
    board: PuzzleBoard
    solutions: list[list[PuzzleMove]]

    @property
    def id(self) -> int:
        return self.puzzleId


def dict_to_puzzle(data: dict):
    return Puzzle(
        puzzleId=data['puzzleId'],
        difficulty=PuzzleDifficulty(data['difficulty']),
        board=PuzzleBoard(**data['board']),
        solutions=[
            [PuzzleMove(**move_data) for move_data in solution]
            for solution in data['solutions']
        ]
    )

def load_puzzles(filepath: str):
    with open(filepath, 'r') as f:
        raw_list = json.load(f)  # load the entire JSON array
    puzzles = [dict_to_puzzle(item) for item in raw_list]
    return puzzles


# 0-indexed grid indexing to AlphaGo board formatting
def array_index_format_to_game_format(row: int, col: int):
    true_row = row + 1
    true_col = col + 1

    return true_row, true_col

# puzzle json indexing to AlphaGo board formatting (1-7 cols L2R and (6-1 rows T2B))
def puzzle_index_format_to_game_format(row: int, col: int, num_rows: int):
    true_row = num_rows - row + 1
    true_col = col

    return true_row, true_col


def puzzle_to_gamestate(game_name: str, puzzle: Puzzle):
    num_rows = puzzle.board.num_rows
    num_cols = puzzle.board.num_cols

    board = Board(num_rows=num_rows, num_cols=num_cols)
    gs = GameState.new_game(board_size=(num_rows, num_cols))
    # Populate board
    for row in reversed(range(num_rows)):
        for col in range(num_cols):
            player_at_board_pos = puzzle.board.grid[row][col]
            if player_at_board_pos == 0:
                continue

            player = Player(player_at_board_pos)
            true_row, true_col = array_index_format_to_game_format(row, col)
            point = Point(row=true_row, col=true_col)
            board.place_stone(player, point)

    next_player_val = puzzle.solutions[0][0].player
    next_player = Player(next_player_val)
    gs = GameState(board=board, next_player=next_player, previous=None, move=None)

    return gs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="connectFour")  # The game name, which should also be the encoder name of that game.
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')
    parser.add_argument('--verbose', action="store_true")

    args = parser.parse_args()

    game_name = args.game
    agent_filename = args.agent
    device = args.device
    verbose = args.verbose

    puzzles_path = "SimplePuzzles.json"
    puzzles = load_puzzles(puzzles_path)

    agent = rl.ACAgent.load(agent_filename, Model, device=device)

    moves_correct = 0
    total_moves = 0

    puzzles_correct = 0


    for i, puzzle in enumerate(puzzles):
        if verbose:
            print(f"Puzzle {i}:")
        game_state = puzzle_to_gamestate(game_name, puzzle)

        puzzle_solved = True

        for sol_move_num in range(len(puzzle.solutions[0])):
            # Computer's turn at uneven solution move numbers
            if sol_move_num % 2 == 1:
                puzzle_move = puzzle.solutions[0][sol_move_num]
                r, c = puzzle_index_format_to_game_format(puzzle_move.row, puzzle_move.column, puzzle.board.num_rows)
                move = Move(
                    Point(
                        row=r,
                        col=c,
                    )
                )
                game_state = game_state.apply_move(move)
                continue

            if verbose:
                print(f"Solution move {sol_move_num+1}:")
                game_state.board.visualize()

            unique_moves: List[PuzzleMove] = []
            for solution_path in puzzle.solutions:
                solution_move = solution_path[sol_move_num]
                if solution_move not in unique_moves:
                    unique_moves.append(solution_move)
            correct_move_prob_per_move = 1 / len(unique_moves)
            correct_cols = [puzzle_index_format_to_game_format(m.row, m.column, puzzle.board.num_rows)[1] for m in unique_moves]

            if verbose:
                print(f"Found {len(unique_moves)} unique moves for move number {sol_move_num+1}")
                print("-----------------------------------------------------")
                print("Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7")
                print("-----------------------------------------------------")
                # Get solution move probabilities and print
                probs = [
                    f"{correct_move_prob_per_move:.3f}" if col in correct_cols else f"{0:.3f}"
                    for col in range(1, 8)
                ]
                print(" | ".join(probs))

            # Get agent move probabilities and print
            move_probs, _ = move_logits = agent.predict_policy_and_value(game_state=game_state)
            probs_2d = move_probs.reshape(
                puzzle.board.num_rows,
                puzzle.board.num_cols,
            )
            max_col_probs = torch.max(probs_2d, dim=0).values

            if verbose:
                max_prob_strings = [f"{prob:.3f}" for prob in max_col_probs]
                print(" | ".join(max_prob_strings))
                print("-----------------------------------------------------")

            # Make player move
            puzzle_move = puzzle.solutions[0][sol_move_num]
            r, c = puzzle_index_format_to_game_format(puzzle_move.row, puzzle_move.column, puzzle.board.num_rows)
            move = Move(
                Point(
                    row=r,
                    col=c,
                )
            )
            game_state = game_state.apply_move(move)

            agent_answer = int(torch.argmax(max_col_probs))+1
            if agent_answer in correct_cols:
                moves_correct += 1
                total_moves += 1
            else:
                puzzle_solved = False
                total_moves += 1

        if verbose:
            print("Solved!")
            game_state.board.visualize()
            print("\n\n")

        if puzzle_solved:
            puzzles_correct += 1

    print(f"Final statistics for agent {agent_filename}:")
    print(f"Move accuracy: {moves_correct/total_moves:.3f}%")
    print(f"Puzzle accuracy: {puzzles_correct/len(puzzles):.3f}%")
