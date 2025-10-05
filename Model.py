import torch.nn as nn
from DLCF import encoders

class Model(nn.Module):
    def __init__(self, encoder: encoders.Encoder):
        super(Model, self).__init__()

        self.hidden_size = 512
        self.in_dim = encoder.num_planes

        flatten_size = 64 * encoder.board_height * encoder.board_width

        self.backbone = nn.Sequential(
            *[
                # Layer 1
                nn.Conv2d(in_channels=self.in_dim, out_channels=64, kernel_size=(3, 3), padding='same'),
                nn.ReLU(),
                # Layer 2
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
                nn.ReLU(),
                # Layer 3
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
                nn.ReLU(),
                # Flatten and project
                nn.Flatten(),
                nn.Linear(flatten_size, self.hidden_size),
            ]
        )

        self.policy = nn.Sequential(
            *[
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, encoder.num_points()),
                nn.Softmax(dim=-1),
            ]
        )

        self.value = nn.Sequential(
            *[
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
                nn.Tanh(),
            ]
        )

    def forward(self, encoded_board):
        processed_board = self.backbone(encoded_board)

        policy_ouput = self.policy(processed_board)
        value_output = self.value(processed_board)

        return policy_ouput, value_output
