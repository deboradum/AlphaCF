import argparse
import h5py

import torch.nn as nn

from DLCF import rl
from DLCF import encoders
from typing import OrderedDict


class Model(nn.Module):
    def __init__(self, hidden_size: int, encoder: encoders.Encoder):
        self.backbone = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Sequential([nn.Conv2d(out_channels=64, kernel_size=(3, 3), padding='same'), nn.ReLU()])),
                    ("conv2", nn.Sequential([nn.Conv2d(out_channels=64, kernel_size=(3, 3), padding='same'), nn.ReLU()])),
                    ("conv3", nn.Sequential([nn.Conv2d(out_channels=64, kernel_size=(3, 3), padding='same'), nn.ReLU()])),
                    ("flat", nn.Flatten()),
                    ("processed_board", nn.Linear(FLATTEN_SIZE, hidden_size)),
                ]
            )
        )

        self.policy = nn.Sequential(
            OrderedDict(
                [
                    ("policy_hidden_layer", nn.Sequential([nn.Linear(hidden_size, hidden_size), nn.ReLU()])),
                    ("policy_output", nn.Sequential([nn.Linear(hidden_size, self.encoder.num_points()), nn.Softmax()])),
                ]
            )
        )

        self.value = nn.Sequential(
            OrderedDict(
                [
                    ("value_hidden_layer", nn.Sequential([nn.Linear(hidden_size, hidden_size), nn.ReLU()])),
                    ("value_output", nn.Sequential([nn.Linear(hidden_size, self.encoder.num_points()), nn.Softmax()])),
                ]
            )
        )

    def forward(self, encoded_board):
        processed_board = self.backbone(encoded_board)

        policy_ouput = self.policy(processed_board)
        value_output = self.value(processed_board)

        return policy_ouput, value_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--output-file')
    parser.add_argument('--encoder-name', type=str, default="connectFour")
    HIDDEN_SIZE = 512
    args = parser.parse_args()

    board_size = args.board_size
    output_file = args.output_file


    encoder = encoders.get_encoder_by_name(args.encoder_name, board_size)
    model = Model(HIDDEN_SIZE, encoder)

    new_agent = rl.ACAgent(model, encoder)
    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)

if __name__ == '__main__':
    main()
