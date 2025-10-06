import argparse
import h5py
import multiprocessing as mp
from DLCF import rl
from tqdm import tqdm
from Model import Model
from typing import Tuple
from DLCF.rl import ACAgent
from functools import partial
from collections import namedtuple
from DLCF.cfBoard import GameState, Player
from constants import WIN_REWARD, LOSS_REWARD


class GameRecord(namedtuple('GameRecord', 'winner')):
    pass


def simulate_game(black_player: ACAgent, white_player: ACAgent, board_size: Tuple[int, int], verbose:bool=False):
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    winner = game.compute_game_result()

    if verbose:
        game.board.visualize()
        print("Winner is player: ", winner)
        print()

    return GameRecord(winner=winner)


def run_game_worker(game_idx: int, agent_filename: str, board_size: Tuple[int, int]):
    """
    Worker function that simulates a single game and returns the experience.
    'game_idx' is just a placeholder to make each call to the worker unique.
    """
    agent1 = rl.load_ac_agent(h5py.File(agent_filename, 'r'), Model)
    agent2 = rl.load_ac_agent(h5py.File(agent_filename, 'r'), Model)

    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()

    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    collector1.begin_episode()
    collector2.begin_episode()

    # Note: simulate_game now needs board_size passed to it
    game_record = simulate_game(agent1, agent2, board_size)

    if game_record.winner == Player.black:
        collector1.complete_episode(reward=WIN_REWARD)
        collector2.complete_episode(reward=LOSS_REWARD)
    else:
        collector2.complete_episode(reward=WIN_REWARD)
        collector1.complete_episode(reward=LOSS_REWARD)

    # Return the collected experience for this single game
    return collector1, collector2


def selfPlayMultiThreaded(agent_filename: str, experience_filename: str, num_games: int, board_size: Tuple[int, int], num_workers: int):
    # Use partial to create a new function with some arguments pre-filled.
    # This is needed because pool.map only accepts one iterable argument.
    worker_func = partial(run_game_worker, agent_filename=agent_filename, board_size=board_size)

    all_collectors = []

    # Create the pool of worker processes
    # Using 'with' ensures the pool is properly closed
    with mp.Pool(processes=num_workers) as pool:
        # Use tqdm to show progress as tasks are completed by the pool
        # pool.imap_unordered is often more responsive for progress bars
        game_indices = range(num_games)
        pbar = tqdm(pool.imap_unordered(worker_func, game_indices), total=num_games, desc="Generating experience")

        for collector1, collector2 in pbar:
            all_collectors.append(collector1)
            all_collectors.append(collector2)

    # 3. Combine the experience from all the workers
    experience = rl.combine_experience(all_collectors)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def selfPlay(agent_filename: str, experience_filename: str, num_games: int, board_size: Tuple[int, int]):
    agent1 = rl.load_ac_agent(h5py.File(agent_filename), Model)
    agent2 = rl.load_ac_agent(h5py.File(agent_filename), Model)

    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()

    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for _ in tqdm(range(num_games), desc="Generating experience"):
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2, board_size)

        if game_record.winner == Player.black:
            collector1.complete_episode(reward=WIN_REWARD)
            collector2.complete_episode(reward=LOSS_REWARD)
        else:
            collector2.complete_episode(reward=WIN_REWARD)
            collector1.complete_episode(reward=LOSS_REWARD)

    experience = rl.combine_experience([collector1, collector2])
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7], help="The board size as (heigth, width) (default., 6 7)")
    parser.add_argument('--learning-agent', type=str, required=True)
    parser.add_argument('--experience-out', type=str, required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)

    args = parser.parse_args()

    agent_filename = args.learning_agent
    experience_filename = args.experience_out
    num_games = args.num_games
    board_size = args.board_size

    selfPlay(agent_filename, experience_filename, num_games, tuple(board_size))
