import torch
import argparse
from DLCF import rl
from tqdm import tqdm
from Model import Model
from typing import Tuple
from DLCF.rl import ACAgent
from DLCF.DLCFtypes import Player
from collections import namedtuple
from DLCF.getGameState import getGameState

from constants import WIN_REWARD, LOSS_REWARD


class GameRecord(namedtuple('GameRecord', 'winner')):
    pass


def simulate_game(game_name: str, black_player: ACAgent, white_player: ACAgent, board_size: Tuple[int, int], verbose:bool=False):
    game = getGameState(game_name=game_name).new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    with torch.no_grad():
        while not game.is_over():
            next_move = agents[game.next_player].select_move(game)
            game = game.apply_move(next_move)

    winner = game.compute_game_result()

    if verbose:
        game.board.visualize()
        print("Winner is player: ", winner)
        print()

    return GameRecord(winner=winner)


def selfPlay(game_name: str, agent_filename: str, experience_filename: str, num_games: int, board_size: Tuple[int, int], device: str = "cpu", worker_id: int = 0):
    agent1 = rl.ACAgent.load(agent_filename, Model, device=device)
    agent2 = rl.ACAgent.load(agent_filename, Model, device=device)

    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()

    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for _ in tqdm(range(num_games), desc=f"Generating experience (worker {worker_id})", position=worker_id):
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(game_name, agent1, agent2, board_size)

        if game_record.winner == Player.black:
            collector1.complete_episode(reward=WIN_REWARD)
            collector2.complete_episode(reward=LOSS_REWARD)
        elif game_record.winner == Player.white:
            collector2.complete_episode(reward=WIN_REWARD)
            collector1.complete_episode(reward=LOSS_REWARD)
        else: # Draw
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)

    experience = rl.combine_experience([collector1, collector2])
    experience.save(experience_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="connectFour")  # The game name, which should also be the encoder name of that game.
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7], help="The board size as (heigth, width) (default., 6 7)")
    parser.add_argument('--learning-agent', type=str, required=True)
    parser.add_argument('--experience-out', type=str, required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')

    args = parser.parse_args()

    game_name = args.game
    agent_filename = args.learning_agent
    experience_filename = args.experience_out
    num_games = args.num_games
    board_size = args.board_size
    device = args.device

    selfPlay(game_name, agent_filename, experience_filename, num_games, tuple(board_size), device=device)
