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
    def __init__(self, model: Agent, encoder: Encoder, device: str = "cpu"):
        self._model: Agent = model
        self._encoder = encoder
        self._collector = None
        self.device = device

    def set_collector(self, collector: ExperienceCollector):
        self._collector = collector

    def gamestate_value(self, game_state: GameState):
        X = self._encoder.encode(game_state)
        _, values = self._model(X.unsqueeze(0).to(self.device))
        estimated_value = values.item()

        return estimated_value

    def sample_move(self, game_state: GameState):
        num_moves = self._encoder.board_width * self._encoder.board_height

        X = self._encoder.encode(game_state)

        policy_logits, values = self._model(X.unsqueeze(0).to(self.device))
        estimated_value = values.item()

        valid_move_mask = torch.zeros(num_moves, dtype=torch.bool, device=self.device)
        for move in game_state.legal_moves():
            # Convert each valid Move object back to its corresponding index
            move_idx = self._encoder.encode_point(move.point)
            valid_move_mask[move_idx] = True

        masked_logits = policy_logits.squeeze(0).clone()
        masked_logits[~valid_move_mask] = float('-inf')

        move_probs = nn.functional.softmax(masked_logits, dim=-1)

        point_idx = torch.multinomial(move_probs, 1).item()

        return X, point_idx, estimated_value

    def select_move(self, game_state: GameState):
        X, point_idx, estimated_value = self.sample_move(game_state)

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

        total_policy_loss = 0
        total_value_loss = 0
        total_combined_loss = 0
        num_batches = 0

        for states_batch, actions_batch, advantages_batch, value_target_batch in tqdm(data_loader, desc="Training agent"):
            states_batch = states_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)
            advantages_batch = advantages_batch.to(self.device)
            value_target_batch = value_target_batch.to(self.device)

            optimizer.zero_grad()

            policy_logits, value_pred = self._model(states_batch)

            log_policy_preds = nn.functional.log_softmax(policy_logits, dim=-1)
            selected_log_probs = log_policy_preds[torch.arange(len(actions_batch)), actions_batch]

            # Policy loss: -E[advantage * log(pi(action|state))]
            policy_loss = -torch.mean(advantages_batch * selected_log_probs)
            value_loss = value_loss_fn(value_pred, value_target_batch)
            total_loss = policy_loss + 0.5 * value_loss

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_combined_loss += total_loss.item()
            num_batches += 1

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1)
            optimizer.step()

        return (total_policy_loss / num_batches,  total_value_loss / num_batches,  total_combined_loss / num_batches)

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

    def save(self, path: str):
        torch.save({
            'encoder_config': {
                'name': self._encoder.name(),
                'board_width': self._encoder.board_width,
                'board_height': self._encoder.board_height,
            },
            'model_state_dict': self._model.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, model_class: torch.nn.Module, device: str):
        data = torch.load(path, map_location=device, weights_only=False)

        encoder_config = data['encoder_config']
        encoder = get_encoder_by_name(
            encoder_config['name'],
            (encoder_config['board_height'], encoder_config['board_width'])
        )

        model = model_class(encoder)
        model.load_state_dict(data['model_state_dict'])
        model.to(device)

        return cls(model, encoder, device=device)
