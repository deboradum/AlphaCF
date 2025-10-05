import os
import argparse

from typing import Tuple
from selfPlay import selfPlay
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
    if old_agent_path is None:
        agent_base = "./agents/newAgent/"
        os.makedirs(agent_base, exist_ok=True)

        old_agent_path = f"{agent_base}/gen1"
        initAgent(board_size=board_size, encoder_name=encoder_name, output_file=old_agent_path)

    current_generation = 1

    experience_base_path = "./agents/newAgent/experiences"
    os.makedirs(experience_base_path, exist_ok=True)
    gen_experiences = []
    gen_iteration = 0
    while current_generation < num_generations+1:
        experience_filepath = f"{experience_base_path}/gen1_{gen_iteration}"
        selfPlay(
            agent_filename=old_agent_path,
            experience_filename=experience_filepath,
            num_games=num_games_per_iteration,
            board_size=board_size
        )
        gen_experiences.append(experience_filepath)

        new_agent_path = f"./agents/newAgent/gen{current_generation}"
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
        if win_rate_agent_1 < 0.45:
            current_generation += 1
            gen_experiences = []
            gen_iteration = 0
            print(f"New agent was better after {gen_iteration} iterations. Going to generation {current_generation} now.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=False)
    parser.add_argument('--encoder-name', type=str, default="connectFour")
    parser.add_argument('--num-generations', type=int, default=10)
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
