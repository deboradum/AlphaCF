import os
import torch
import wandb
import random
import argparse
import shutil
from typing import Tuple

from mctsEvalAgent import mctsEvalAgent
from mctsSelfPlay import mctsSelfPlay
from mctsTrainAgent import mctsTrainAgent


def MCTSimprove(
    game_name: str,
    board_size: Tuple[int, int],
    base_agent_path: str,
    num_generations: int,
    num_games_per_iteration: int,
    play_batch_size: int,
    learning_rate: float,
    train_batch_size: int,
    ppo_epochs: int,
    mcts_num_rounds: int,
    mcts_c_puct: float,
    mcts_temperature: float,
    device: str,
    verbose: bool = False,
):
    assert os.path.isfile(base_agent_path)
    old_agent_path = base_agent_path

    base_agent_path_dir = os.path.dirname(base_agent_path)
    experience_base_path = f"{base_agent_path_dir}/mctsExperiences"
    os.makedirs(experience_base_path, exist_ok=True)

    current_generation = 0
    gen_iteration = 0
    last_agents = [old_agent_path]
    num_experiences = 0
    current_lr = learning_rate
    current_entropy_coef = entropy_coef

    while current_generation < num_generations:
        opponent_path = old_agent_path

        experience_filepath = f"{experience_base_path}/gen{current_generation}_{gen_iteration}"

        mctsSelfPlay(
            game_name=game_name,
            agent_filename=opponent_path,
            experience_filename=experience_filepath,
            num_games=num_games_per_iteration,
            board_size=board_size,
            batch_size=play_batch_size,
            num_rounds=mcts_num_rounds,
            c_puct=mcts_c_puct,
            temperature=mcts_temperature,
            device="cpu" if device == "mps" else device,  # Self play is faster on cpu than mps
        )

        new_agent_path = f"{base_agent_path_dir}/mctsGen{current_generation+1}"
        policy_loss, entropy_loss, value_loss, total_loss, grad_norm_before_clip, grad_norm_after_clip = mctsTrainAgent(
            learning_agent_filename=old_agent_path,
            experience_files=[experience_filepath],
            updated_agent_filename=new_agent_path,
            learning_rate=learning_rate,
            batch_size=train_batch_size,
            entropy_coef=0,
            ppo_epochs=ppo_epochs,
            clip_epsilon=0.0,
            device=device
        )

        win_rate_new_agent = mctsEvalAgent(
            game_name=game_name,
            agent1_path=new_agent_path,
            agent2_path=old_agent_path,
            num_games=512,  # tmp
            board_size=board_size,
            batch_size=play_batch_size,
            num_rounds=mcts_num_rounds,
            c_puct=mcts_c_puct,
            device="cpu" if device == "mps" else device,  # eval is faster on cpu than mps
            verbose=verbose,
        )

        num_experiences += num_games_per_iteration
        wandb.log({
            "win_rate": win_rate_new_agent,
            "generation": current_generation,
            "current_learing_rate": current_lr,
            "current_entropy_coefficient": current_entropy_coef,
            "iteration": gen_iteration,
            "total_experiences": num_experiences,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
            "grad_norm_before_clip": grad_norm_before_clip,
            "grad_norm_after_clip": grad_norm_after_clip,
        })

        gen_iteration += 1
        # If new netowrk is better
        if win_rate_new_agent > 0.5:
            old_agent_path = new_agent_path
            current_generation += 1

            last_agents.append(new_agent_path)
            if len(last_agents) > 15:
                oldest = last_agents.pop(0)
                gen_num = int(oldest.split("gen")[-1])
                if gen_num % 10 != 0 and os.path.exists(oldest):
                    os.remove(oldest)

            for exp_file in os.listdir(experience_base_path):
                os.remove(os.path.join(experience_base_path, exp_file))

            print(f"\nNew agent was better after {gen_iteration} iterations. Going to generation {current_generation} now.")
            gen_iteration = 0
            current_lr *= learning_rate_decay
            current_entropy_coef *= learning_rate_decay
        else:
            if os.path.exists(new_agent_path):
                os.remove(new_agent_path)

        # If agent is not better after 3 iterations, model is either locally optimal or too heavily overfitted
        if gen_iteration > 4:
            for exp_file in os.listdir(experience_base_path):
                os.remove(os.path.join(experience_base_path, exp_file))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")
    parser.add_argument('--base-agent', type=str, required=True, help="Path to the pre-trained PPO agent.")
    parser.add_argument('--num-generations', type=int, default=100)
    parser.add_argument('--num-games-per-iteration', type=int, default=10000)
    parser.add_argument('--play-bs', type=int, default=50, help="Number of games to simulate in parallel during self-play.")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-decay', type=float, default=0.99)
    parser.add_argument('--train-bs', type=int, default=1024)
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--ppo-epochs', type=int, default=3)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7])
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--verbose', action="store_true")

    parser.add_argument('--mcts-rounds', type=int, default=200, help="Number of MCTS simulations per move.")
    parser.add_argument('--mcts-c-puct', type=float, default=1.0, help="MCTS exploration constant.")
    parser.add_argument('--mcts-temp', type=float, default=1.0, help="MCTS temperature for move selection.")
    args = parser.parse_args()

    game_name = args.game
    base_agent_path = args.base_agent
    num_generations = args.num_generations
    num_games_per_iteration = args.num_games_per_iteration
    play_batch_size = args.play_bs
    learning_rate = args.lr
    learning_rate_decay = args.lr_decay
    train_batch_size = args.train_bs
    entropy_coef = args.entropy_coef
    ppo_epochs = args.ppo_epochs
    clip_epsilon = args.clip_epsilon
    board_size = args.board_size
    device = args.device
    seed = args.seed
    verbose = args.verbose

    mcts_rounds = args.mcts_rounds
    mcts_c_puct = args.mcts_c_puct
    mcts_temp = args.mcts_temp

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    assert os.path.isfile(base_agent_path), f"Base agent not found at {base_agent_path}"

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        device = 'cpu'

    board_size_str = f"{board_size[0]}x{board_size[1]}"

    wandb.init(
        project=f"Alpha{game_name}",
        name=f"{base_agent_path.split('/')[-2]}-mcts",
        config={
            "seed": seed,
            "encoder": game_name,
            "learning_rate": learning_rate,
            "learning_rate_decay": learning_rate_decay,
            "ppo_epochs": ppo_epochs,
            "clip_epsilon": clip_epsilon,
            "train_batch_size": train_batch_size,
            "generations": num_generations,
            "games_per_iteration": num_games_per_iteration,
            "play_batch_size": play_batch_size,
            "board_size": board_size,
            "device": device,

            "mcts_rounds": mcts_rounds,
            "mcts_c_puct": mcts_c_puct,
            "mcts_temp": mcts_temp,
        },
        tags=[board_size_str, "mcts"],
    )

    MCTSimprove(
        game_name=args.game,
        board_size=tuple(args.board_size),
        base_agent_path=args.base_agent,
        num_generations=args.num_generations,
        num_games_per_iteration=args.num_games_per_iteration,
        play_batch_size=args.play_bs,
        learning_rate=args.lr,
        train_batch_size=args.train_bs,
        ppo_epochs=args.ppo_epochs,
        mcts_num_rounds=args.mcts_rounds,
        mcts_c_puct=args.mcts_c_puct,
        mcts_temperature=args.mcts_temp,
        device=args.device,
        verbose=args.verbose,
    )

    wandb.finish()
