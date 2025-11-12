import argparse

from tqdm import tqdm
from Model import Model
from typing import Tuple
from DLCF.rl import ACAgent
from DLCF.DLCFtypes import Player
from selfPlay import simulate_games


def evalAgent(game_name: str, agent1_path: str, agent2_path: str, num_games: int, board_size: Tuple[int, int], batch_size: int, device: str = "cpu", verbose:bool = False):
    assert num_games > batch_size, "Need more eval games than batch size"
    num_batched_simualtions = num_games // batch_size

    agent1 = ACAgent.load(agent1_path, Model, device=device)
    agent2 = ACAgent.load(agent2_path, Model, device=device)

    wins = 0
    losses = 0
    color1 = Player.black
    for _ in tqdm(range(num_batched_simualtions), desc="Evaluating agent against old version"):
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2

        game_records = simulate_games(game_name, black_player, white_player, board_size, batch_size, verbose=verbose)
        wins += sum(1 if game_record.winner == color1 else 0 for game_record in game_records)
        losses += sum(1 if game_record.winner == color1.other else 0 for game_record in game_records)

        color1 = color1.other
    print(f'Agent 1 record: {wins}/{wins + losses} ({wins / (wins+losses):.3f})%')

    return wins / (wins + losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")  # The game name, which should also be the encoder name of that game.
    parser.add_argument('--batch-size', '-b', type=int, default=512)
    parser.add_argument('--agent1', required=True)
    parser.add_argument('--agent2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7], help="The board size as (heigth, width) (default., 6 7)")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')
    parser.add_argument('--verbose', action="store_true")

    args = parser.parse_args()

    game_name = args.game
    batch_size = args.batch_size
    agent1_path = args.agent1
    agent2_path = args.agent2
    num_games = args.num_games
    board_size = args.board_size
    device = args.device

    evalAgent(
        game_name=game_name,
        agent1_path=agent1_path,
        agent2_path=agent2_path,
        num_games=num_games,
        board_size=board_size,
        batch_size=batch_size,
        device=device,
        verbose=args.verbose
    )
