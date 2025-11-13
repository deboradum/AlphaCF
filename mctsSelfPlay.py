import torch
import argparse
from DLCF import rl
from tqdm import tqdm
from Model import Model
from typing import Tuple, List
from DLCF.DLCFtypes import Player
from collections import namedtuple
from DLCF.mcts.mcts import MCTSAgent
from DLCF.getGameState import getGameState
from constants import WIN_REWARD, LOSS_REWARD


class GameRecord(namedtuple('GameRecord', 'winner')):
    pass


def simulate_game(
    game_name: str,
    black_player: MCTSAgent,
    white_player: MCTSAgent,
    board_size: Tuple[int, int],
):
    agents = [
        black_player,
        white_player,
    ]

    game = getGameState(game_name=game_name).new_game(board_size)

    turn_idx = 0
    with torch.no_grad():
        while True:
            selected_moves = agents[turn_idx].select_moves([game])
            selected_move = selected_moves[0]  # bs 1

            # Draw
            if selected_move is None:
                winner = None
                break

            game = game.apply_move(selected_move)
            turn_idx = (turn_idx + 1) % 2

            if game.is_over():
                winner = game.compute_game_result()
                break

    return GameRecord(winner=winner)


def mctsSelfPlay(
    game_name: str,
    agent_filename: str,
    experience_filename: str,
    num_games: int,
    board_size: Tuple[int, int],
    batch_size: int,
    num_rounds: int,
    c_puct: float,
    temperature: float,
    device: str = "cpu"
):
    ac_agent1 = rl.ACAgent.load(agent_filename, Model, device=device)
    mcts_agent1 = MCTSAgent(
        ac_agent=ac_agent1,
        num_rounds=num_rounds,
        c_puct=c_puct,
        temperature=temperature
    )
    ac_agent2 = rl.ACAgent.load(agent_filename, Model, device=device)
    mcts_agent2 = MCTSAgent(
        ac_agent=ac_agent2,
        num_rounds=num_rounds,
        c_puct=c_puct,
        temperature=temperature
    )


    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()

    mcts_agent1.ac_agent.set_collectors([collector1])
    mcts_agent2.ac_agent.set_collectors([collector2])

    for _ in tqdm(range(num_games), desc=f"Generating experience"):
        for i in range(batch_size):
            collector1.begin_episode()
            collector2.begin_episode()

        game_record = simulate_game(game_name, mcts_agent1, mcts_agent2, board_size)

        if game_record.winner == Player.black:
            collector1.complete_episode(reward=WIN_REWARD)
            collector2.complete_episode(reward=LOSS_REWARD)
        elif game_record.winner == Player.white:
            collector2.complete_episode(reward=WIN_REWARD)
            collector1.complete_episode(reward=LOSS_REWARD)
        else: # Draw
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)

    experience = rl.combine_experience([[collector1], [collector2]])
    experience.save(experience_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")
    parser.add_argument('--agent', type=str, required=True, help="Path to the base ACAgent model.")
    parser.add_argument('--experience-out', type=str, required=True, help="Path to save the experience buffer.")
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--batch-size', '-b', type=int, default=10, help="Number of games to simulate in parallel (or sequentially in this version).")
    parser.add_argument('--buffer-size', type=int, default=20000, help="Total size of the experience replay buffer.")
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7])
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu')

    parser.add_argument('--num-rounds', type=int, default=100, help="Number of MCTS simulations per move.")
    parser.add_argument('--c-puct', type=float, default=1.0, help="Exploration constant.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Controls move selection stochasticity.")

    args = parser.parse_args()

    mctsSelfPlay(
        game_name=args.game,
        agent_filename=args.agent,
        experience_filename=args.experience_out,
        num_games=args.num_games,
        board_size=tuple(args.board_size),
        batch_size=args.batch_size,
        num_rounds=args.num_rounds,
        c_puct=args.c_puct,
        temperature=args.temperature,
        device=args.device
    )
