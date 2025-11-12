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
    mcts_agent: MCTSAgent,
    board_size: Tuple[int, int],
    collector: MCTSExperienceCollector
):
    ac_agent = mcts_agent.ac_agent
    encoder = ac_agent._encoder
    num_moves = encoder.num_points()

    game = getGameState(game_name=game_name).new_game(board_size)
    collector.begin_episode()

    while True:
        policy_target, selected_move = mcts_agent.run_search(game)

        if selected_move is None:
            collector.complete_episode(winner_val=0.0)
            break

        state_tensor = encoder.encode([game])[0]
        collector.record_decision(state=state_tensor, policy_target=policy_target)

        game = game.apply_move(selected_move)

        if game.is_over():
            winner = game.compute_game_result()
            if winner == Player.black:
                winner_val = WIN_REWARD
            elif winner == Player.white:
                winner_val = LOSS_REWARD
            else:
                winner_val = 0.0

            collector.complete_episode(winner_val=winner_val)
            break
    return


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
    assert num_games > batch_size, "Need more eval games than batch size"

    num_batches = num_games // batch_size

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


    collectors1 = [rl.ExperienceCollector() for _ in range(batch_size)]
    collectors2 = [rl.ExperienceCollector() for _ in range(batch_size)]

    mcts_agent1.ac_agent.set_collectors(collectors1)
    mcts_agent2.ac_agent.set_collectors(collectors2)

    for _ in tqdm(range(num_batched_simualtions), desc=f"Generating experience"):
        for i in range(batch_size):
            collectors1[i].begin_episode()
            collectors2[i].begin_episode()

        game_records = simulate_games(game_name, mcts_agent1, mcts_agent2, board_size, batch_size=batch_size)

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

    try:
        experience_buffer = MCTSExperienceBuffer.load(args.experience_out, args.buffer_size)
        print(f"Loaded existing experience buffer from {args.experience_out}. Size: {len(experience_buffer)}")
    except FileNotFoundError:
        experience_buffer = MCTSExperienceBuffer(buffer_size=args.buffer_size)
        print(f"Created new experience buffer. Size: {args.buffer_size}")
    except Exception as e:
        print(f"Could not load buffer, creating new one. Error: {e}")
        experience_buffer = MCTSExperienceBuffer(buffer_size=args.buffer_size)

    mctsSelfPlay(
        game_name=args.game,
        agent_filename=args.agent,
        experience_buffer=experience_buffer,
        num_games=args.num_games,
        board_size=tuple(args.board_size),
        batch_size=args.batch_size,
        num_rounds=args.num_rounds,
        c_puct=args.c_puct,
        temperature=args.temperature,
        device=args.device
    )

    experience_buffer.save(args.experience_out)
    print(f"Experience saved to {args.experience_out}")
