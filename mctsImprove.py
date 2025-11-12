import os
import torch
import wandb
import random
import argparse
import shutil
from typing import Tuple

from Model import Model
from initAgent import initAgent
from evalAgent import evalAgent
from mctsSelfPlay import mctsSelfPlay
from mctsTrainAgent import mctsTrainAgent
from DLCF.rl.mctsExperience import MCTSExperienceBuffer

def trainMCTS(
    game_name: str,
    agent_name: str,
    encoder_name: str,
    board_size: Tuple[int, int],
    base_agent_path: str,
    num_generations: int,
    num_games_per_gen: int,
    play_batch_size: int,
    buffer_size: int,
    learning_rate: float,
    train_batch_size: int,
    train_epochs: int,
    l2_reg: float,
    mcts_num_rounds: int,
    mcts_c_puct: float,
    mcts_temperature: float,
    eval_games: int,
    device: str,
    verbose: bool = False,
):
    agent_base = f"./agents/{agent_name}_MCTS"
    os.makedirs(agent_base, exist_ok=True)

    gen_files = [f for f in os.listdir(agent_base) if f.startswith('gen') and f[3:].isdigit()]
    if gen_files:
        highest_gen_num = max([int(f[3:]) for f in gen_files])
        current_agent_path = f"{agent_base}/gen{highest_gen_num}"
        current_generation = highest_gen_num
        print(f"Resuming MCTS training from generation {current_generation}")
    else:
        current_agent_path = f"{agent_base}/gen0"
        shutil.copy(base_agent_path, current_agent_path)
        current_generation = 0
        print(f"Starting MCTS training. Copied base agent from {base_agent_path} to {current_agent_path}")

    experience_path = f"{agent_base}/experience_buffer.pth"
    if os.path.exists(experience_path):
        experience_buffer = MCTSExperienceBuffer.load(experience_path, buffer_size)
        print(f"Loaded {len(experience_buffer)} samples from existing buffer.")
    else:
        experience_buffer = MCTSExperienceBuffer(buffer_size=buffer_size)
        print("Created new, empty experience buffer.")

    while current_generation < num_generations:
        gen_num = current_generation + 1
        mctsSelfPlay(
            game_name=game_name,
            agent_filename=current_agent_path,
            experience_buffer=experience_buffer,
            num_games=num_games_per_gen,
            board_size=board_size,
            batch_size=play_batch_size,
            num_rounds=mcts_num_rounds,
            c_puct=mcts_c_puct,
            temperature=mcts_temperature,
            device=device
        )

        experience_buffer.save(experience_path)

        if len(experience_buffer) < train_batch_size:
            print(f"Not enough samples in buffer ({len(experience_buffer)}) to meet batch size ({train_batch_size}). Skipping training.")
            current_generation += 1 # We still count this as a gen to gather more data
            continue

        new_agent_path = f"{agent_base}/gen{gen_num}_temp"

        policy_loss, value_loss, total_loss = mctsTrainAgent(
            learning_agent_filename=current_agent_path,
            experience_buffer=experience_buffer,
            updated_agent_filename=new_agent_path,
            learning_rate=learning_rate,
            batch_size=train_batch_size,
            train_epochs=train_epochs,
            l2_reg=l2_reg,
            device=device
        )

        win_rate_new_vs_old = evalAgent(
            game_name=game_name,
            agent1_path=new_agent_path,       # New agent
            agent2_path=current_agent_path,   # Old agent
            num_games=eval_games,
            board_size=board_size,
            batch_size=play_batch_size,
            device=device,
            verbose=verbose,
        )

        print(f"Evaluation: New Agent Win Rate vs Old: {win_rate_new_vs_old * 100:.2f}%")

        wandb.log({
            "generation": gen_num,
            "win_rate_vs_prev_gen": win_rate_new_vs_old,
            "total_experiences": len(experience_buffer),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
        })

        if win_rate_new_vs_old > 0.55:
            final_new_path = f"{agent_base}/gen{gen_num}"
            os.rename(new_agent_path, final_new_path)
            current_agent_path = final_new_path
        else:
            os.remove(new_agent_path)

        current_generation += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")
    parser.add_argument('--agent-name', type=str, default="AlphaZeroAgent", help="Name for the new MCTS agent directory.")
    parser.add_argument('--base-agent', type=str, required=True, help="Path to the pre-trained PPO agent (e.g., agents/my_ppo_agent/gen100).")
    parser.add_argument('--num-generations', type=int, default=100)
    parser.add_argument('--num-games-per-gen', type=int, default=500)
    parser.add_argument('--play-bs', type=int, default=50, help="Number of games to simulate in parallel during self-play.")
    parser.add_argument('--buffer-size', type=int, default=50000, help="Max number of (s, pi, z) tuples in replay buffer.")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--train-bs', type=int, default=1024)
    parser.add_argument('--train-epochs', type=int, default=3)
    parser.add_argument('--l2-reg', type=float, default=1e-4)
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7])
    parser.add_argument('--eval-games', type=int, default=100, help="Number of games to play for evaluation.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--verbose', action="store_true")

    parser.add_argument('--mcts-rounds', type=int, default=200, help="Number of MCTS simulations per move.")
    parser.add_argument('--mcts-c-puct', type=float, default=1.0, help="MCTS exploration constant.")
    parser.add_argument('--mcts-temp', type=float, default=1.0, help="MCTS temperature for move selection.")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.base_agent):
        print(f"Error: Base agent not found at {args.base_agent}")
        exit(1)

    wandb.init(
        project=f"Alpha-{args.game}",
        name=f"{args.agent_name}-MCTS",
        config=vars(args),
    )

    trainMCTS(
        game_name=args.game,
        agent_name=args.agent_name,
        encoder_name=args.game,
        board_size=tuple(args.board_size),
        base_agent_path=args.base_agent,
        num_generations=args.num_generations,
        num_games_per_gen=args.num_games_per_gen,
        play_batch_size=args.play_bs,
        buffer_size=args.buffer_size,
        learning_rate=args.lr,
        train_batch_size=args.train_bs,
        train_epochs=args.train_epochs,
        l2_reg=args.l2_reg,
        mcts_num_rounds=args.mcts_rounds,
        mcts_c_puct=args.mcts_c_puct,
        mcts_temperature=args.mcts_temp,
        eval_games=args.eval_games,
        device=args.device,
        verbose=args.verbose,
    )

    wandb.finish()
