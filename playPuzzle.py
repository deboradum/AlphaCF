import time
import json
import torch
import argparse

from DLCF import rl
from enum import Enum
from Model import Model
from typing import List
from dataclasses import dataclass
from DLCF.mcts.mcts import MCTSAgent
from DLCF.DLCFtypes import Player, Move, Point
from DLCF.connectFourBoard import Board, GameState
import numpy as np

np.set_printoptions(linewidth=200, precision=8, suppress=True)


class PuzzleDifficulty(str, Enum):
    SIMPLE = "Simple"
    TOUGH = "Tough"
    HARD = "Hard"
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


def print_debug_data(game_state: GameState, unique_moves: List, sol_move_num: int, solution_prob_strings: List[str], max_prob_strings: List[str], value_strings: List[str]):
    print(f"Solution move {sol_move_num+1}:")
    print("Board after move:")
    game_state.board.visualize()

    print(f"Found {len(unique_moves)} unique winning moves for move number {sol_move_num+1}")
    print("---------------------------------------------------------------------------")
    print("               Col 1  | Col 2  | Col 3  | Col 4  | Col 5  | Col 6  | Col 7 ")
    print("---------------------------------------------------------------------------")
    print("True probs:  | ", end="")
    print(" | ".join(solution_prob_strings))

    print("Pred probs:  | ", end="")
    print(" | ".join(max_prob_strings))

    print("Move values: | ", end="")
    print(" | ".join(value_strings))
    print("---------------------------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="connectFour")  # The game name, which should also be the encoder name of that game.
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')
    parser.add_argument('--mcts', action="store_true")
    parser.add_argument('--num-rounds', type=int, default=500, help="Number of MCTS simulations per move for eval.")
    parser.add_argument('--c-puct', type=float, default=1.0, help="MCTS exploration constant.")
    parser.add_argument('--verbose', action="store_true")

    args = parser.parse_args()

    game_name = args.game
    agent_filename = args.agent
    device = args.device
    verbose = args.verbose

    puzzles_path = "SimplePuzzles.json"
    puzzles = load_puzzles(puzzles_path)

    agent = rl.ACAgent.load(agent_filename, Model, device=device)
    if args.mcts:
        agent = MCTSAgent(ac_agent=agent, num_rounds=args.num_rounds, c_puct=args.c_puct, eval_mode=True)

    moves_correct = 0
    total_moves = 0
    puzzles_correct = 0

    s = time.time()
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

            # Get solution move probabilities
            unique_moves: List[PuzzleMove] = []
            for solution_path in puzzle.solutions:
                solution_move = solution_path[sol_move_num]
                if solution_move not in unique_moves:
                    unique_moves.append(solution_move)
            correct_move_prob_per_move = 1 / len(unique_moves)
            correct_cols = [puzzle_index_format_to_game_format(m.row, m.column, puzzle.board.num_rows)[1] for m in unique_moves]

            # Get agent move probabilities and post-move values
            game_states_to_predict = [game_state]
            col_to_batch_index = {}
            legal_moves = game_state.legal_moves()
            for move in legal_moves:
                col_idx = move.point.col - 1 # 0-indexed
                next_state = game_state.apply_move(move)
                col_to_batch_index[col_idx] = len(game_states_to_predict)
                game_states_to_predict.append(next_state)

            # AC agent
            if not args.mcts:
                move_probs, estimated_values = agent.predict_policy_and_value(game_states=game_states_to_predict)
                # We only need move probs for the current (first) game state
                current_state_probs = move_probs[0]
            # MCTS agent
            else:
                _, _, _, _, _, _, current_state_probs, current_state_val, _ = agent._run_search(game_state)

            probs_2d = current_state_probs.reshape(
                puzzle.board.num_rows,
                puzzle.board.num_cols,
            )
            max_col_probs = torch.max(probs_2d, dim=0).values

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

            # Print board and predicted/ true solution data
            if verbose:
                solution_prob_strings = [f"{correct_move_prob_per_move:+.3f}" if col in correct_cols else f"{0:+.3f}"for col in range(1, 8)]
                max_prob_strings = [f"{prob:+.3f}" for prob in max_col_probs]
                hypothetical_values_by_col = [0.0] * puzzle.board.num_cols
                # for col_idx, batch_idx in col_to_batch_index.items():
                #     hypothetical_values_by_col[col_idx] = estimated_values[batch_idx].item()
                # value_strings = [f"{v:+.3f}" for v in hypothetical_values_by_col]
                value_strings = [f"{0:+.3f}"]*7

                print_debug_data(game_state, unique_moves, sol_move_num, solution_prob_strings, max_prob_strings, value_strings)

            # Record stats
            agent_answer = int(torch.argmax(max_col_probs))+1
            if agent_answer in correct_cols:
                moves_correct += 1
                total_moves += 1
            else:
                puzzle_solved = False
                total_moves += 1

        if verbose:
            print("Solved!" if puzzle_solved else f"Puzzle failed.")
            print("\n\n")

        if puzzle_solved:
            puzzles_correct += 1

    taken = time.time() - s
    print(f"Final statistics for agent {agent_filename}:")
    print(f"Move accuracy: {moves_correct/total_moves:.3f}%")
    print(f"Puzzle accuracy: {puzzles_correct/len(puzzles):.3f}%")
    print(f"Took {taken:.2f}s")
