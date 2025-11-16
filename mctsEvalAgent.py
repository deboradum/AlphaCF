import math
import argparse
from tqdm import tqdm
from Model import Model
from typing import Tuple
from DLCF.mcts.mcts import MCTSAgent
from mctsSelfPlay import simulate_batch
from DLCF.rl import ACAgent, ExperienceCollector

class DummyCollector(ExperienceCollector):
    def begin_episode(self): pass
    def record_decision(self, **kwargs): pass
    def complete_episode(self, **kwargs): pass

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
    base_agent1 = ACAgent.load(agent1_path, Model, device=device)
    mcts_agent1 = MCTSAgent(base_agent1, num_rounds=num_rounds, c_puct=c_puct, temperature=0.0, eval_mode=True)

    base_agent2 = ACAgent.load(agent2_path, Model, device=device)
    mcts_agent2 = MCTSAgent(base_agent2, num_rounds=num_rounds, c_puct=c_puct, temperature=0.0, eval_mode=True)

    total_wins_agent1 = 0
    total_wins_agent2 = 0
    total_draws = 0

    class ResultCollector(ExperienceCollector):
        def __init__(self):
            self.last_reward = 0
        def begin_episode(self):
            self.last_reward = 0
        def record_decision(self, **kwargs): pass
        def complete_episode(self, reward):
            self.last_reward = reward

    games_as_black = num_games // 2
    games_as_white = num_games - games_as_black

    batches_as_black = math.ceil(games_as_black / batch_size)
    batches_as_white = math.ceil(games_as_white / batch_size)

    print(f"Eval: Agent 1 playing Black for {games_as_black} games, White for {games_as_white} games.")

    if games_as_black > 0:
        collectors_black = [ResultCollector() for _ in range(batch_size)]
        collectors_white = [ResultCollector() for _ in range(batch_size)]

        for _ in tqdm(range(batches_as_black), desc="Agent 1 (Black) vs Agent 2 (White)"):
            current_batch_size = min(batch_size, games_as_black)
            games_as_black -= current_batch_size

            # Reset collectors
            for c in collectors_black[:current_batch_size]: c.begin_episode()
            for c in collectors_white[:current_batch_size]: c.begin_episode()

            simulate_batch(
                game_name=game_name,
                black_player=mcts_agent1,
                white_player=mcts_agent2,
                board_size=board_size,
                batch_size=current_batch_size,
                black_collectors=collectors_black[:current_batch_size],
                white_collectors=collectors_white[:current_batch_size]
            )

            for i in range(current_batch_size):
                r = collectors_black[i].last_reward
                if r > 0: total_wins_agent1 += 1
                elif r < 0: total_wins_agent2 += 1
                else: total_draws += 1

    if games_as_white > 0:
        collectors_black = [ResultCollector() for _ in range(batch_size)]
        collectors_white = [ResultCollector() for _ in range(batch_size)]

        for _ in tqdm(range(batches_as_white), desc="Agent 2 (Black) vs Agent 1 (White)"):
            current_batch_size = min(batch_size, games_as_white)
            games_as_white -= current_batch_size

            for c in collectors_black[:current_batch_size]: c.begin_episode()
            for c in collectors_white[:current_batch_size]: c.begin_episode()

            simulate_batch(
                game_name=game_name,
                black_player=mcts_agent2,
                white_player=mcts_agent1,
                board_size=board_size,
                batch_size=current_batch_size,
                black_collectors=collectors_black[:current_batch_size],
                white_collectors=collectors_white[:current_batch_size]
            )

            for i in range(current_batch_size):
                r = collectors_white[i].last_reward
                if r > 0: total_wins_agent1 += 1
                elif r < 0: total_wins_agent2 += 1
                else: total_draws += 1

    total_played = total_wins_agent1 + total_wins_agent2 + total_draws
    win_rate_agent1 = total_wins_agent1 / total_played if total_played > 0 else 0

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
    parser.add_argument('--batch-size', '-b', type=int, default=10, help="Parallel games.")
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
