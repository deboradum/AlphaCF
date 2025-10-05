import h5py
import torch
import torch.nn as nn

from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

from DLCF import goboard
from DLCF.agent import Agent
from DLCF.agent.helpers import is_point_an_eye
from DLCF.encoders import Encoder, get_encoder_by_name
from DLCF.rl.experience import ExperienceBuffer, ExperienceCollector

class ACAgent(Agent):
    def __init__(self, model, encoder: Encoder):
        self._model = model
        self._encoder = encoder
        self._collector = None

    def set_collector(self, collector: ExperienceCollector):
        self._collector = collector

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height

        X = self._encoder.encode(game_state)

        actions, values = self._model(X)
        move_probs = actions[0]
        estimated_value = values[0][0]

        eps = 1e-6
        move_probs = torch.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / torch.sum(move_probs)

        candidates = torch.arange(num_moves)
        ranked_moves = torch.random.choice(
            candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            fills_own_eye = is_point_an_eye(
                game_state.board, point,
                game_state.next_player)
            if move_is_valid and (not fills_own_eye):
                if self._collector is not None:
                    self._collector.record_decision(
                        state=X,
                        action=point_idx,
                        estimated_value=estimated_value)
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

    def train(self, experience: ExperienceBuffer, lr:float=0.1, batch_size:int=128):
        self._model.train()

        optimizer = SGD(self._model.parameters(), lr=lr)
        value_loss_fn = nn.MSELoss()

        states_tensor = experience.states
        actions_tensor = experience.actions
        advantages_tensor = experience.advantages
        rewards_tensor = experience.rewards

        value_target = rewards_tensor.view(-1, 1)

        dataset = TensorDataset(states_tensor, actions_tensor, advantages_tensor, value_target)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for states_batch, actions_batch, advantages_batch, value_target_batch in data_loader:
            optimizer.zero_grad()

            policy_pred, value_pred = self._model(states_batch)

            # Policy loss: -E[advantage * log(pi(action|state))]
            log_policy_preds = torch.log(torch.clamp(policy_pred, 1e-12))
            selected_log_probs = log_policy_preds[torch.arange(len(actions_batch)), actions_batch]
            policy_loss = -1 * torch.mean(advantages_batch * selected_log_probs)

            value_loss = value_loss_fn(value_pred, value_target_batch)

            total_loss = policy_loss + 0.5 * value_loss

            total_loss.backward()
            optimizer.step()

    def serialize(self, h5file):
        encoder_group = h5file.create_group('encoder')
        model_group = h5file.create_group('model')

        # Save encoder attributes
        encoder_group.attrs['name'] = self._encoder.name()
        encoder_group.attrs['board_width'] = self._encoder.board_width
        encoder_group.attrs['board_height'] = self._encoder.board_height

        # Save model state_dict
        model_state_dict = self._model.state_dict()
        for key, tensor in model_state_dict.items():
            model_group.create_dataset(key, data=tensor.cpu().numpy())


def load_ac_agent(h5file: h5py.File, model_class: torch.nn.Module):
    """Loads an agent from an HDF5 file."""
    # Load encoder attributes
    encoder_group = h5file['encoder']
    encoder_name = encoder_group.attrs['name']
    board_width = encoder_group.attrs['board_width']
    board_height = encoder_group.attrs['board_height']

    # Recreate the encoder
    encoder = get_encoder_by_name(
        encoder_name,
        (board_width, board_height)
    )

    model = model_class(encoder)

    # Reconstruct the state_dict from the HDF5 file
    model_group = h5file['model']
    state_dict = {}
    for key, dataset in model_group.items():
        state_dict[key] = torch.from_numpy(dataset[()])

    # Load the weights into the model
    model.load_state_dict(state_dict)

    # Create and return the new agent
    return ACAgent(model, encoder)
