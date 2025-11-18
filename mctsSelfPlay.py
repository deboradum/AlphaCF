import math
import torch
import argparse
from DLCF import rl
from tqdm import tqdm
from Model import Model
from typing import Tuple, List
from DLCF.DLCFtypes import Player
from DLCF.mcts.mcts import MCTSAgent
from DLCF.getGameState import getGameState
from constants import WIN_REWARD, LOSS_REWARD

def simulate_batch(
    game_name: str,
    black_player: MCTSAgent,
    white_player: MCTSAgent,
    board_size: Tuple[int, int],
    batch_size: int,
    black_collectors: List[rl.ExperienceCollector],
    white_collectors: List[rl.ExperienceCollector]
):
    games = [getGameState(game_name=game_name).new_game(board_size) for _ in range(batch_size)]
    active_indices = list(range(batch_size))

    with torch.no_grad():
        while active_indices:
            first_game = games[active_indices[0]]
            next_p = first_game.next_player

            if next_p == Player.black:
                current_agent = black_player
                current_collectors = [black_collectors[i] for i in active_indices]
            else:
                current_agent = white_player
                current_collectors = [white_collectors[i] for i in active_indices]

            current_agent.ac_agent.set_collectors(current_collectors)

            active_games = [games[i] for i in active_indices]
            selected_moves = current_agent.select_moves(active_games)

            next_active_indices = []

            for i, move in enumerate(selected_moves):
                original_idx = active_indices[i]
                game = games[original_idx]

                if move is None:
                    # Draw
                    black_collectors[original_idx].complete_episode(reward=0)
                    white_collectors[original_idx].complete_episode(reward=0)
                    continue

                game = game.apply_move(move)
                games[original_idx] = game

                if game.is_over():
                    winner = game.compute_game_result()

                    if winner == Player.black:
                        black_collectors[original_idx].complete_episode(reward=WIN_REWARD)
                        white_collectors[original_idx].complete_episode(reward=LOSS_REWARD)
                    elif winner == Player.white:
                        white_collectors[original_idx].complete_episode(reward=WIN_REWARD)
                        black_collectors[original_idx].complete_episode(reward=LOSS_REWARD)
                    else: # Draw
                        black_collectors[original_idx].complete_episode(reward=0)
                        white_collectors[original_idx].complete_episode(reward=0)
                else:
                    next_active_indices.append(original_idx)

            active_indices = next_active_indices


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
    mcts_agent1 = MCTSAgent(ac_agent1, num_rounds=num_rounds, c_puct=c_puct, temperature=temperature)

    ac_agent2 = rl.ACAgent.load(agent_filename, Model, device=device)
    mcts_agent2 = MCTSAgent(ac_agent2, num_rounds=num_rounds, c_puct=c_puct, temperature=temperature)

    collectors_black = [rl.ExperienceCollector() for _ in range(batch_size)]
    collectors_white = [rl.ExperienceCollector() for _ in range(batch_size)]

    num_batches = math.ceil(num_games / batch_size)

    for _ in tqdm(range(num_batches), desc="Generating experience"):
        for c in collectors_black: c.begin_episode()
        for c in collectors_white: c.begin_episode()

        simulate_batch(
            game_name=game_name,
            black_player=mcts_agent1,
            white_player=mcts_agent2,
            board_size=board_size,
            batch_size=batch_size,
            black_collectors=collectors_black,
            white_collectors=collectors_white
        )

    all_collectors = [collectors_black, collectors_white]
    experience = rl.combine_experience(all_collectors)
    experience.save(experience_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")
    parser.add_argument('--agent', type=str, required=True, help="Path to the base ACAgent model.")
    parser.add_argument('--experience-out', type=str, required=True, help="Path to save the experience buffer.")
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--batch-size', '-b', type=int, default=10, help="Number of games to simulate in parallel.")
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7])
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu')

    parser.add_argument('--num-rounds', type=int, default=100, help="Number of MCTS simulations per move.")
    parser.add_argument('--c-puct', type=float, default=1.0, help="Exploration constant.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Controls move selection stochasticity.")

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
