import torch
import random
import torch.nn as nn
from tqdm import tqdm

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from DLCF import cfBoard
from DLCF.DLCFtypes import Player
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

        self.optimizer = None

    def set_collector(self, collector: ExperienceCollector):
        self._collector = collector

    def gamestate_value(self, game_state: GameState):
        X = self._encoder.encode(game_state)
        _, values = self._model(X.unsqueeze(0).to(self.device))
        estimated_value = values.item()

        return estimated_value

    def debug_move(self, game_state: GameState, move_probs: torch.Tensor, point_idx: int):
        print("\n--- DEBUG MOVE ---")

        # Reshape probabilities to match the board dimensions for easier reading
        board_probs = move_probs.detach().numpy().reshape(
            (self._encoder.board_height, self._encoder.board_width)
        )

        print("Policy Network Move Probabilities:")
        print("Black: 'X', White: 'O'")
        print('Black (X)' if game_state.next_player == Player.black else 'White (O)', "to play")
        # Print formatted probabilities for each column
        for row_idx in range(self._encoder.board_height):
            row_items = []
            for col_idx in range(self._encoder.board_width):
                # Point objects are 1-indexed in cfBoard
                point = cfBoard.Point(row=row_idx + 1, col=col_idx + 1)
                player = game_state.board.get(point)

                if player is not None:
                    # If a stone is on the board, show X or O
                    stone = '  X  ' if player == Player.black else '  O  '
                    row_items.append(stone)
                else:
                    # Otherwise, show the move probability
                    prob = board_probs[row_idx, col_idx]
                    row_items.append(f"{prob:.3f}")

            print(f"| {' | '.join(row_items)} |")

        selected_point = self._encoder.decode_point_index(point_idx)
        selected_prob = move_probs[point_idx].item()

        print(f"\nSelected Move: Play in column {selected_point.col}")
        print(f"Probability of Selected Move: {selected_prob:.4f}")
        print("------------------\n")

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

        debug = False
        if debug:
            self.debug_move(game_state, move_probs, point_idx)

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

    def train(self, experience: ExperienceBuffer, lr:float=0.0001, batch_size:int=128, entropy_coef: float = 0.001):
        self._model.train()

        if self.optimizer is None:
            self.optimizer = Adam(self._model.parameters(), lr=lr)
        value_loss_fn = nn.MSELoss()

        states_tensor = experience.states
        actions_tensor = experience.actions
        advantages_tensor = experience.advantages
        rewards_tensor = experience.rewards

        value_target = rewards_tensor.view(-1, 1)

        dataset = TensorDataset(states_tensor, actions_tensor, advantages_tensor, value_target)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0
        total_entropy_loss = 0
        total_value_loss = 0
        total_combined_loss = 0
        total_grad_norm = 0
        num_batches = 0

        for states_batch, actions_batch, advantages_batch, value_target_batch in tqdm(data_loader, desc="Training agent"):
            states_batch = states_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)
            advantages_batch = advantages_batch.to(self.device)
            value_target_batch = value_target_batch.to(self.device)
            # Normalize advantages for better
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.optimizer.zero_grad()

            policy_logits, value_pred = self._model(states_batch)
            log_policy_preds = nn.functional.log_softmax(policy_logits, dim=-1)
            selected_log_probs = log_policy_preds[torch.arange(len(actions_batch)), actions_batch]

            # Encourage entropy to prevent overconfident predicitons (mainly needed early on)
            policy_probs = log_policy_preds.exp()
            entropy = -(policy_probs * log_policy_preds).sum(dim=1)

            # Policy loss: -E[advantage * log(pi(action|state))]
            policy_loss = -torch.mean(advantages_batch * selected_log_probs)
            entropy_loss = -entropy_coef * entropy.mean()
            value_loss = value_loss_fn(value_pred, value_target_batch)
            # L = -E[advantage * log(\pi(action|state))] + \Beta H(\pi(\cdot|state)) + 0.5 * MSE(\hat{V}, V)
            total_loss = policy_loss + entropy_loss + (0.5 * value_loss)

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_value_loss += value_loss.item()
            total_combined_loss += total_loss.item()
            total_grad_norm += grad_norm
            num_batches += 1


        return (
            total_policy_loss / num_batches,
            total_entropy_loss / num_batches,
            total_value_loss / num_batches,
            total_combined_loss / num_batches,
            total_grad_norm / num_batches,
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
