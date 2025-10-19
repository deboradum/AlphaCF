import torch
import argparse
from DLCF import rl
from tqdm import tqdm
from Model import Model
from typing import Tuple, List
from DLCF.rl import ACAgent
from DLCF.DLCFtypes import Player, GameStateTemplate
from collections import namedtuple
from DLCF.getGameState import getGameState

from constants import WIN_REWARD, LOSS_REWARD


class GameRecord(namedtuple('GameRecord', 'winner')):
    pass


def simulate_games(game_name: str, black_player: ACAgent, white_player: ACAgent, board_size: Tuple[int, int], batch_size: int, verbose: bool=False):
    games: List[GameStateTemplate] = [getGameState(game_name=game_name).new_game(board_size) for _ in range(batch_size)]
    agents = [
        black_player,
        white_player,
    ]

    num_done = 0
    turn_idx = 0
    with torch.no_grad():
        while not num_done == batch_size:
            next_moves = agents[turn_idx].select_moves(games)
            num_done = sum([1 if g.is_over() else 0 for g in games])
            games = [g.apply_move(m) if not g.is_over() else g for g, m in zip(games, next_moves)]
            turn_idx = (turn_idx + 1) % 2

    winners = [game.compute_game_result() for game in games]

    if verbose:
        black_winners = winners.count(Player.black)
        white_winners = winners.count(Player.white)
        draws = batch_size - black_winners - white_winners
        print(f"Number of black winners: {winners.count(Player.black)} ({winners.count(Player.black)/batch_size:.2f}%)")
        print(f"Number of white winners: {winners.count(Player.white)} ({winners.count(Player.white)/batch_size:.2f}%)")
        print(f"Number of draws: {draws} ({draws/batch_size:.2f}%)")
        print()

    return [GameRecord(winner=w) for w in winners]


def selfPlay(game_name: str, agent_filename: str, experience_filename: str, num_games: int, board_size: Tuple[int, int], batch_size: int, device: str = "cpu", verbose: bool=False):
    assert num_games > batch_size, "Need more eval games than batch size"
    num_batched_simualtions = num_games // batch_size

    agent1 = rl.ACAgent.load(agent_filename, Model, device=device)
    agent2 = rl.ACAgent.load(agent_filename, Model, device=device)

    collectors1 = [rl.ExperienceCollector() for _ in range(batch_size)]
    collectors2 = [rl.ExperienceCollector() for _ in range(batch_size)]

    agent1.set_collectors(collectors1)
    agent2.set_collectors(collectors2)

    for _ in tqdm(range(num_batched_simualtions), desc=f"Generating batched experience"):
        for i in range(batch_size):
            collectors1[i].begin_episode()
            collectors2[i].begin_episode()

        game_records = simulate_games(game_name, agent1, agent2, board_size, batch_size=batch_size, verbose=verbose)

        for i in range(batch_size):
            if game_records[i].winner == Player.black:
                collectors1[i].complete_episode(reward=WIN_REWARD)
                collectors2[i].complete_episode(reward=LOSS_REWARD)
            elif game_records[i].winner == Player.white:
                collectors2[i].complete_episode(reward=WIN_REWARD)
                collectors1[i].complete_episode(reward=LOSS_REWARD)
            else: # Draw
                collectors1[i].complete_episode(reward=0)
                collectors2[i].complete_episode(reward=0)

    experience = rl.combine_experience([collectors1, collectors2])
    experience.save(experience_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="connectFour")  # The game name, which should also be the encoder name of that game.
    parser.add_argument('--batch-size', '-b', type=int, default=512)
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7], help="The board size as (heigth, width) (default., 6 7)")
    parser.add_argument('--learning-agent', type=str, required=True)
    parser.add_argument('--experience-out', type=str, required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')

    args = parser.parse_args()

    game_name = args.game
    batch_size = args.batch_size
    agent_filename = args.learning_agent
    experience_filename = args.experience_out
    num_games = args.num_games
    board_size = args.board_size
    device = args.device

    selfPlay(game_name, agent_filename, experience_filename, num_games, tuple(board_size), batch_size= batch_size, device=device)
