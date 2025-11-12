import torch
import torch.nn as nn
from tqdm import tqdm

from typing import List
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from Model import Model
from DLCF.agent import Agent
from DLCF.DLCFtypes import Move
from DLCF.encoders import Encoder, get_encoder_by_name
from DLCF.getGameState import GameStateTemplate
from DLCF.rl.experience import ExperienceBuffer, ExperienceCollector

class ACAgent(Agent):
    def __init__(self, model: Model, encoder: Encoder, device: str = "cpu"):
        self._model = model
        self._encoder = encoder
        self._collectors = None
        self.device = device

        self.optimizer = None

    def set_collectors(self, collectors: List[ExperienceCollector]):
        self._collectors = collectors

    def sample_moves(self, game_states: List[GameStateTemplate]):
        num_moves = self._encoder.board_width * self._encoder.board_height
        batch_size = len(game_states)

        Xs = self._encoder.encode(game_states).to(self.device)  # (B, C, H, W)

        policy_logits, values = self._model(Xs)  # (B, num_moves) and (B, 1)
        estimated_values = values.squeeze(-1) # (B,)

        valid_move_mask = torch.zeros(
            batch_size,
            num_moves,
            dtype=torch.bool,
            device=self.device
        )
        batch_indices = []
        move_indices = []
        for i, game_state in enumerate(game_states):
            for move in game_state.legal_moves():
                move_idx = self._encoder.encode_point(move.point)
                batch_indices.append(i)
                move_indices.append(move_idx)

        if batch_indices:
            valid_move_mask[batch_indices, move_indices] = True

        # Prevent 0 probability for games where no valid moves exist
        is_game_over = ~valid_move_mask.any(dim=1)
        if is_game_over.any():
            valid_move_mask[is_game_over, 0] = True

        masked_logits = policy_logits.clone()
        masked_logits[~valid_move_mask] = float('-inf')

        move_probs = nn.functional.softmax(masked_logits, dim=-1)  # (B, num_moves)

        dist = torch.distributions.Categorical(probs=move_probs)
        action_tensors = dist.sample()  # (B,)
        log_probs = dist.log_prob(action_tensors)  # (B,)

        return Xs, action_tensors, estimated_values, log_probs, valid_move_mask

    def select_moves(self, game_states: List[GameStateTemplate]) -> List[Move]:
        bs = len(game_states)

        Xs, action_tensors, estimated_values, log_probs, valid_move_mask = self.sample_moves(game_states)

        if self._collectors is not None:
            action_indices = action_tensors.tolist()
            log_prob_list = log_probs.tolist()
            value_list = estimated_values.tolist()

            for i in range(bs):
                self._collectors[i].record_decision(
                    state=Xs[i],
                    action=action_indices[i],
                    log_prob=log_prob_list[i],
                    estimated_value=value_list[i],
                    mask=valid_move_mask[i]
                )

        point_indices = action_tensors.tolist()
        moves = [
            Move.play(self._encoder.decode_point_index(idx))
            for idx in point_indices
        ]

        return moves

    def predict_policy_and_value(self, game_states: List[GameStateTemplate]):
        num_moves = self._encoder.board_width * self._encoder.board_height
        bs = len(game_states)

        Xs = self._encoder.encode(game_states).to(self.device)  # (B, C, H, W)

        policy_logits, values = self._model(Xs)  # (B, num_moves) and (B, 1)
        estimated_values = values.squeeze(-1) # (B,)

        valid_move_mask = torch.zeros(
            bs,
            num_moves,
            dtype=torch.bool,
            device=self.device
        )
        batch_indices = []
        move_indices = []
        for i, game_state in enumerate(game_states):
            for move in game_state.legal_moves():
                move_idx = self._encoder.encode_point(move.point)
                batch_indices.append(i)
                move_indices.append(move_idx)

        if batch_indices:
            valid_move_mask[batch_indices, move_indices] = True

        # Prevent 0 probability for games where no valid moves exist
        is_game_over = ~valid_move_mask.any(dim=1)
        if is_game_over.any():
            valid_move_mask[is_game_over, 0] = True

        masked_logits = policy_logits.clone()
        masked_logits[~valid_move_mask] = float('-inf')

        move_probs = nn.functional.softmax(masked_logits, dim=-1)  # (B, num_moves)

        return move_probs, estimated_values


    def train(self, experience: ExperienceBuffer, lr:float=0.0001, batch_size:int=128, entropy_coef: float = 0.001, ppo_epochs: int = 3, clip_epsilon: float = 0.2):
        self._model.train()

        if self.optimizer is None:
            self.optimizer = Adam(self._model.parameters(), lr=lr)
        value_loss_fn = nn.MSELoss()

        dataset = TensorDataset(
            experience.states,
            experience.actions,
            experience.advantages,
            experience.rewards,
            experience.old_log_probs,
            experience.masks,
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_combined_loss = 0

        total_grad_norm_before = 0
        total_grad_norm_after = 0
        num_batches = 0

        for _ in tqdm(range(ppo_epochs), desc="Training agent"):
            for states_batch, actions_batch, advantages_batch, value_target_batch, old_log_probs_batch, masks_batch in data_loader:
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)
                advantages_batch = advantages_batch.to(self.device)
                value_target_batch = value_target_batch.to(self.device).view(-1, 1)
                old_log_probs_batch = old_log_probs_batch.to(self.device)
                masks_batch = masks_batch.to(self.device)

                # Normalize advantages for better
                std = advantages_batch.std()
                if std > 1e-8:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / std
                else:
                    advantages_batch = advantages_batch - advantages_batch.mean()

                self.optimizer.zero_grad()

                policy_logits, value_pred = self._model(states_batch)

                # Mask illegal moves
                is_game_over = ~masks_batch.any(dim=1)
                if is_game_over.any():
                    masks_batch[is_game_over, 0] = True
                masked_logits = policy_logits.clone()
                masked_logits[~masks_batch] = float('-inf')

                dist = torch.distributions.Categorical(logits=masked_logits)

                new_log_probs = dist.log_prob(actions_batch)

                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = value_loss_fn(value_pred, value_target_batch)

                entropy_loss = -entropy_coef * dist.entropy().mean()

                total_loss = policy_loss + 0.5 * value_loss + entropy_loss

                total_loss.backward()
                grad_norm_before = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=0.5)
                grad_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in self._model.parameters() if p.grad is not None))
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_combined_loss += total_loss.item()

                total_grad_norm_before += grad_norm_before
                total_grad_norm_after += grad_norm_after
                num_batches += 1

        return (
            total_policy_loss / num_batches,
            total_entropy_loss / num_batches,
            total_value_loss / num_batches,
            total_combined_loss / num_batches,
            total_grad_norm_before / num_batches,
            total_grad_norm_after / num_batches,
        )

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
