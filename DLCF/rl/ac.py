import h5py
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

from DLCF import cfBoard
from DLCF.agent import Agent
from DLCF.cfBoard import GameState
from DLCF.encoders import Encoder, get_encoder_by_name
from DLCF.rl.experience import ExperienceBuffer, ExperienceCollector

class ACAgent(Agent):
    def __init__(self, model, encoder: Encoder):
        self._model = model
        self._encoder = encoder
        self._collector = None

    def set_collector(self, collector: ExperienceCollector):
        self._collector = collector

    def select_move(self, game_state: GameState):
        num_moves = self._encoder.board_width * self._encoder.board_height

        X = self._encoder.encode(game_state)

        actions, values = self._model(X.unsqueeze(0))
        move_probs = actions
        estimated_value = values.item()

        valid_move_mask = torch.zeros(num_moves, dtype=torch.bool)
        for move in game_state.legal_moves():
            # Convert each valid Move object back to its corresponding index
            move_idx = self._encoder.encode_point(move.point)
            valid_move_mask[move_idx] = True

        masked_probs = move_probs * valid_move_mask.float()
        if torch.sum(masked_probs) > 0:
            masked_probs = masked_probs / torch.sum(masked_probs)
        else:
            print("This should not happen!")
            # Failsafe: if model gives 0 probability to all valid moves,
            # choose uniformly from the valid moves.
            masked_probs = valid_move_mask.float() / torch.sum(valid_move_mask.float())

        point_idx = torch.multinomial(masked_probs, 1).item()

        if self._collector is not None:
            self._collector.record_decision(
                state=X,
                action=point_idx,
                estimated_value=estimated_value)

        point = self._encoder.decode_point_index(point_idx)
        return cfBoard.Move.play(point)

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

        for states_batch, actions_batch, advantages_batch, value_target_batch in tqdm(data_loader, desc="Training agent"):
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
        (board_height, board_width)
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
