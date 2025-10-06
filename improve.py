import os
import argparse

from typing import Tuple
from selfPlay import selfPlay, selfPlayMultiThreaded
from evalAgent import evalAgent
from initAgent import initAgent
from trainAgent import trainAgent


def improve(
    old_agent_path: str | None,
    encoder_name:str,
    board_size: Tuple[int, int],
    num_generations: int,
    num_games_per_iteration: int,
    learning_rate: float,
    batch_size: int,
    verbose: bool = False,
):
    agent_base = "./agents/newAgent"
    if old_agent_path is None:
        os.makedirs(agent_base, exist_ok=True)

        old_agent_path = f"{agent_base}/gen1"
        initAgent(board_size=board_size, encoder_name=encoder_name, output_file=old_agent_path)

    current_generation = 1

    experience_base_path = f"{agent_base}/experiences"
    os.makedirs(experience_base_path, exist_ok=True)
    gen_experiences = []
    gen_iteration = 0
    while current_generation < num_generations+1:
        experience_filepath = f"{experience_base_path}/gen1_{gen_iteration}"
        selfPlay(
        # selfPlayMultiThreaded(
            agent_filename=old_agent_path,
            experience_filename=experience_filepath,
            num_games=num_games_per_iteration,
            board_size=board_size,
            # num_workers=16,
        )
        gen_experiences.append(experience_filepath)

        new_agent_path = f"{agent_base}/gen{current_generation}"
        trainAgent(
            learning_agent_filename=old_agent_path,
            experience_files=gen_experiences,
            updated_agent_filename=new_agent_path,
            learning_rate=learning_rate,
            batch_size=batch_size
        )

        win_rate_agent_1 = evalAgent(
            agent1_path=old_agent_path,
            agent2_path=new_agent_path,
            num_games=num_games_per_iteration,
            board_size=board_size,
            verbose=verbose,
        )

        gen_iteration += 1
        if win_rate_agent_1 < 0.5:
            previous_agent_path = old_agent_path
            previous_gen_number = current_generation - 1
            old_agent_path = new_agent_path

            if previous_gen_number > 0 and previous_gen_number % 10 != 0:
                if os.path.exists(previous_agent_path):
                    os.remove(previous_agent_path)

            for exp_file in gen_experiences:
                if os.path.exists(exp_file):
                    os.remove(exp_file)

            current_generation += 1
            gen_experiences = []
            print(f"New agent was better after {gen_iteration} iterations. Going to generation {current_generation} now.")
            gen_iteration = 0
        else:
            if os.path.exists(new_agent_path):
                os.remove(new_agent_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=False)
    parser.add_argument('--encoder-name', type=str, default="connectFour")
    parser.add_argument('--num-generations', type=int, default=100)
    parser.add_argument('--num_games_per_iteration', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7], help="The board size as (heigth, width) (default., 6 7)")
    parser.add_argument('--verbose', action="store_true")

    args = parser.parse_args()

    agent_path = args.agent
    encoder_name = args.encoder_name
    num_generations = args.num_generations
    num_games_per_iteration = args.num_games_per_iteration
    learning_rate = args.lr
    batch_size = args.bs
    board_size = args.board_size
    verbose = args.verbose

    improve(
        old_agent_path=agent_path,
        encoder_name=encoder_name,
        board_size=tuple(board_size),
        num_generations=num_generations,
        num_games_per_iteration=num_games_per_iteration,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=verbose,
    )
