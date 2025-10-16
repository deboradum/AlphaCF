import torch.nn as nn
from DLCF.encoders import Encoder


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # The skip connection
        out = self.relu(out)

        return out

class Model(nn.Module):
    def __init__(self, encoder: Encoder):
        super(Model, self).__init__()

        self.num_res_blocks = 2
        self.num_channels = 64
        self.hidden_size = 512
        self.in_dim = encoder.num_planes

        flatten_size = 64 * encoder.board_height * encoder.board_width

        self.initial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_dim,
                out_channels=self.num_channels,
                kernel_size=(3, 3),
                padding='same'
            ),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(self.num_channels) for _ in range(self.num_res_blocks)]
        )

        self.policy = nn.Sequential(
            nn.Conv2d(self.num_channels, 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * encoder.board_height * encoder.board_width, encoder.num_points()),
        )

        self.value = nn.Sequential(
            nn.Conv2d(self.num_channels, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(encoder.board_height * encoder.board_width, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, encoded_board):
        x = self.initial_block(encoded_board)
        x = self.res_blocks(x)

        policy_ouput = self.policy(x)
        value_output = self.value(x)

        return policy_ouput, value_output
