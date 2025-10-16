import argparse

from Model import Model
from typing import Tuple
from DLCF import encoders
from DLCF.rl import ACAgent


def initAgent(board_size: Tuple[int, int], encoder_name: str, output_file: str, device: str = "cpu"):
    encoder = encoders.get_encoder_by_name(encoder_name, board_size)

    model = Model(encoder)
    model.to(device)

    new_agent = ACAgent(model, encoder)
    new_agent.save(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=["ConnectFour", "Gomoku"], default="ConnectFour")  # The game name, which should also be the encoder name of that game.
    parser.add_argument('--board-size', type=int, nargs=2, default=[6, 7], help="The board size as (heigth, width) (default., 6 7)")
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')

    args = parser.parse_args()

    game = args.game
    board_size = args.board_size
    output_file = args.output_file
    device = args.device

    initAgent(tuple(board_size), game, output_file, device=device)
