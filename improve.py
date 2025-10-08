import os
import torch
import wandb
import random
import argparse

from typing import Tuple
from selfPlay import selfPlay
from evalAgent import evalAgent
from initAgent import initAgent
from trainAgent import trainAgent


# TODO: wandb logging
def improve(
    agent_name: str,
    encoder_name: str,
    board_size: Tuple[int, int],
    num_generations: int,
    num_games_per_iteration: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    verbose: bool = False,
):
    agent_base = f"./agents/{agent_name}"
    if not os.path.isdir(agent_base):
        os.makedirs(agent_base, exist_ok=True)
        old_agent_path = f"{agent_base}/gen0"
        initAgent(board_size=board_size, encoder_name=encoder_name, output_file=old_agent_path)
    else:
        gen_files = [f for f in os.listdir(agent_base) if f.startswith('gen') and f[3:].isdigit()]
        if not gen_files:
            # The directory exists but is empty or has no valid 'gen' files
            old_agent_path = f"{agent_base}/gen0"
            initAgent(board_size=board_size, encoder_name=encoder_name, output_file=old_agent_path)
        else:
            highest_gen_num = max([int(f[3:]) for f in gen_files])
            old_agent_path = f"{agent_base}/gen{highest_gen_num}"
            print(f"Resuming agent {agent_base} from generation {highest_gen_num}")


    experience_base_path = f"{agent_base}/experiences"
    os.makedirs(experience_base_path, exist_ok=True)

    gen_experiences = []
    current_generation = int(old_agent_path.split("gen")[-1])
    gen_iteration = 0
    last_agents = [old_agent_path]
    num_experiences = 0

    while current_generation < num_generations:
        # Randomly choose self-play opponent from last 10 networks
        opponent_path = random.choice(last_agents)
        experience_filepath = f"{experience_base_path}/gen{current_generation}_{gen_iteration}"

        selfPlay(
            agent_filename=opponent_path,
            experience_filename=experience_filepath,
            num_games=num_games_per_iteration,
            board_size=board_size,
            device="cpu",
        )
        gen_experiences.append(experience_filepath)

        new_agent_path = f"{agent_base}/gen{current_generation + 1}"
        policy_loss, value_loss, total_loss = trainAgent(
            learning_agent_filename=old_agent_path,
            experience_files=gen_experiences,
            updated_agent_filename=new_agent_path,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
        )

        win_rate_agent_1 = evalAgent(
            agent1_path=old_agent_path,
            agent2_path=new_agent_path,
            num_games=min(num_games_per_iteration, 10000),
            board_size=board_size,
            device="cpu",
            verbose=verbose,
        )

        num_experiences += num_games_per_iteration
        wandb.log({
            "win_rate": 1 - win_rate_agent_1,
            "generation": current_generation,
            "iteration": gen_iteration,
            "total_experiences": num_experiences,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
        })

        gen_iteration += 1
        # If new netowrk is better
        if win_rate_agent_1 < 0.5:
            old_agent_path = new_agent_path
            current_generation += 1
            gen_experiences = []

            last_agents.append(new_agent_path)
            if len(last_agents) > 10:
                oldest = last_agents.pop(0)
                gen_num = int(oldest.split("gen")[-1])
                if gen_num % 10 != 0 and os.path.exists(oldest):
                    os.remove(oldest)

            for exp_file in os.listdir(experience_base_path):
                os.remove(os.path.join(experience_base_path, exp_file))

            print(f"New agent was better after {gen_iteration} iterations. Going to generation {current_generation} now.")
            gen_iteration = 0
        else:
            if os.path.exists(new_agent_path):
                os.remove(new_agent_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default="newAgent")
    parser.add_argument('--encoder-name', type=str, default="connectFour")
    parser.add_argument('--num-generations', type=int, default=100)
    parser.add_argument('--num-games-per-iteration', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7], help="The board size as (heigth, width) (default., 6 7)")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()

    agent_name = args.agent
    encoder_name = args.encoder_name
    num_generations = args.num_generations
    num_games_per_iteration = args.num_games_per_iteration
    learning_rate = args.lr
    batch_size = args.bs
    board_size = args.board_size
    device = args.device
    verbose = args.verbose

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        device = 'cpu'

    wandb.init(
        project="AlphaConnectFour",
        name=agent_name,
        config={
            "agent_name": agent_name,
            "encoder": encoder_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "generations": num_generations,
            "games_per_iteration": num_games_per_iteration,
            "board_size": board_size,
            "device": device,
        }
    )

    improve(
        agent_name=agent_name,
        encoder_name=encoder_name,
        board_size=tuple(board_size),
        num_generations=num_generations,
        num_games_per_iteration=num_games_per_iteration,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    wandb.finish()
