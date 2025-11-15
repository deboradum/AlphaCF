import argparse
from tqdm import tqdm
from Model import Model
from typing import Tuple
from DLCF.rl import ACAgent
from DLCF.DLCFtypes import Player
from DLCF.mcts.mcts import MCTSAgent
from mctsSelfPlay import simulate_game

def mctsEvalAgent(
    game_name: str,
    agent1_path: str,
    agent2_path: str,
    num_games: int,
    board_size: Tuple[int, int],
    num_rounds: int,
    c_puct: float,
    device: str = "cpu",
    verbose: bool = False
):
    base_agent1 = ACAgent.load(agent1_path, Model, device=device)
    mcts_agent1 = MCTSAgent(base_agent1, num_rounds=num_rounds, c_puct=c_puct, temperature=0.0)

    base_agent2 = ACAgent.load(agent2_path, Model, device=device)
    mcts_agent2 = MCTSAgent(base_agent2, num_rounds=num_rounds, c_puct=c_puct, temperature=0.0)

    total_wins_agent1 = 0
    total_wins_agent2 = 0
    total_draws = 0

    for i in tqdm(range(num_games), desc="Evaluating Agents"):
        if i % 2 == 0:
            black_player = mcts_agent1
            white_player = mcts_agent2
            agent1_is_black = True
        else:
            black_player = mcts_agent2
            white_player = mcts_agent1
            agent1_is_black = False

        game_record = simulate_game(
            game_name=game_name,
            black_player=black_player,
            white_player=white_player,
            board_size=board_size
        )

        winner = game_record.winner

        if winner == Player.black:
            if agent1_is_black:
                total_wins_agent1 += 1
            else:
                total_wins_agent2 += 1
        elif winner == Player.white:
            if agent1_is_black:
                total_wins_agent2 += 1
            else:
                total_wins_agent1 += 1
        else:
            total_draws += 1

    total_played = total_wins_agent1 + total_wins_agent2 + total_draws

    win_rate_agent1 = total_wins_agent1 / total_played

    print(f"\n--- Final Results ({total_played} games) ---")
    print(f"Agent 1 wins: {total_wins_agent1} ({win_rate_agent1 * 100:.2f}%)")
    print(f"Agent 2 wins: {total_wins_agent2} ({(total_wins_agent2 / total_played) * 100:.2f}%)")
    print(f"Draws:        {total_draws} ({(total_draws / total_played) * 100:.2f}%)")

    return win_rate_agent1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")
    parser.add_argument('--agent1', required=True)
    parser.add_argument('--agent2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=100)
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
        num_rounds=args.num_rounds,
        c_puct=args.c_puct,
        device=args.device,
        verbose=args.verbose
    )
