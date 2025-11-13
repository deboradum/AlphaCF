import argparse
import torch
from tqdm import tqdm
from Model import Model
from typing import Tuple, List
from DLCF.rl import ACAgent
from DLCF.mcts.mcts import MCTSAgent
from DLCF.DLCFtypes import Player, GameStateTemplate
from DLCF.getGameState import getGameState


def simulate_mcts_games(
    game_name: str,
    agent1: MCTSAgent,
    agent2: MCTSAgent,
    board_size: Tuple[int, int],
    batch_size: int,
    verbose: bool = False
):
    games: List[GameStateTemplate] = [getGameState(game_name=game_name).new_game(board_size) for _ in range(batch_size)]
    agents = [agent1, agent2]

    num_done = 0
    turn_idx = 0

    active_games_indices = list(range(batch_size))
    active_games = [games[i] for i in active_games_indices]

    with torch.no_grad():
        while active_games:
            current_agent = agents[turn_idx]

            next_moves = current_agent.select_moves(active_games)

            new_active_games = []
            new_active_indices = []

            for i, (game_idx, game) in enumerate(zip(active_games_indices, active_games)):
                move = next_moves[i]
                if move is None:
                    continue

                games[game_idx] = game.apply_move(move)

                if not games[game_idx].is_over():
                    new_active_games.append(games[game_idx])
                    new_active_indices.append(game_idx)

            active_games = new_active_games
            active_games_indices = new_active_indices
            turn_idx = (turn_idx + 1) % 2

    winners = [game.compute_game_result() for game in games]

    if verbose:
        black_winners = winners.count(Player.black)
        white_winners = winners.count(Player.white)
        draws = batch_size - black_winners - white_winners
        print(f"Number of black winners (Agent 1): {black_winners} ({black_winners/batch_size:.2f}%)")
        print(f"Number of white winners (Agent 2): {white_winners} ({white_winners/batch_size:.2f}%)")
        print(f"Number of draws: {draws} ({draws/batch_size:.2f}%)")

    return winners.count(Player.black)


def mctsEvalAgent(
    game_name: str,
    agent1_path: str,
    agent2_path: str,
    num_games: int,
    board_size: Tuple[int, int],
    batch_size: int,
    num_rounds: int,
    c_puct: float,
    device: str = "cpu",
    verbose: bool = False
):
    assert num_games >= batch_size, "Need more eval games than batch size"
    num_batched_simulations = num_games // batch_size

    base_agent1 = ACAgent.load(agent1_path, Model, device=device)
    mcts_agent1 = MCTSAgent(base_agent1, num_rounds=num_rounds, c_puct=c_puct, temperature=0.0)

    base_agent2 = ACAgent.load(agent2_path, Model, device=device)
    mcts_agent2 = MCTSAgent(base_agent2, num_rounds=num_rounds, c_puct=c_puct, temperature=0.0)

    total_wins_agent1 = 0
    total_wins_agent2 = 0

    color1 = Player.black
    for _ in tqdm(range(num_batched_simulations), desc="Evaluating MCTS afgent against old version"):
        if color1 == Player.black:
            black_player, white_player = mcts_agent1, mcts_agent2
        else:
            white_player, black_player = mcts_agent1, mcts_agent2

        black_wins = simulate_mcts_games(
            game_name, black_player, white_player, board_size, batch_size, verbose=verbose
        )

        white_wins = batch_size - black_wins

        if color1 == Player.black:
            total_wins_agent1 += black_wins
            total_wins_agent2 += white_wins
        else:
            total_wins_agent1 += white_wins
            total_wins_agent2 += black_wins

        color1 = color1.other

    total_played = total_wins_agent1 + total_wins_agent2
    if total_played == 0:
        print("No games were completed (or all were draws).")
        return 0.0

    win_rate_agent1 = total_wins_agent1 / total_played
    print(f'Agent 1 win rate: {win_rate_agent1 * 100:.2f}% ({total_wins_agent1}/{total_played})')

    return win_rate_agent1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")
    parser.add_argument('--agent1', required=True)
    parser.add_argument('--agent2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--batch-size', '-b', type=int, default=10)
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7])
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--verbose', action="store_true")

    parser.add_argument('--num-rounds', type=int, default=100, help="Number of MCTS simulations per move for eval.")
    parser.add_argument('--c-puct', type=float, default=1.0, help="MCTS exploration constant.")

    args = parser.parse_args()

    mctsEvalAgent(
        game_name=args.game,
        agent1_path=args.agent1,
        agent2_path=args.agent2,
        num_games=args.num_games,
        board_size=tuple(args.board_size),
        batch_size=args.batch_size,
        num_rounds=args.num_rounds,
        c_puct=args.c_puct,
        device=args.device,
        verbose=args.verbose
    )
