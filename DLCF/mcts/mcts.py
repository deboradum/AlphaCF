import math
import torch
import random
import torch.nn as nn
from DLCF import agent
from typing import List
from DLCF.rl import ACAgent
from torch.optim import Adam
from DLCF.DLCFtypes import Move
import torch.nn.functional as F
from DLCF.getGameState import GameStateTemplate
from torch.utils.data import TensorDataset, DataLoader
from DLCF.rl.mctsExperience import MCTSExperienceBuffer

__all__ = [
    'MCTSAgent',
]

class MCTSNode(object):
    def __init__(self, game_state: GameStateTemplate, prob: float, parent: "MCTSNode"= None, move: Move= None):
        self.game_state = game_state
        self.parent = parent
        self.move = move

        self.total_value = 0.
        self.num_rollouts = 0

        self.children: List[MCTSNode] = []
        self.prob = prob  # prior probability p(s,a) from the network

    def backpropagate(self, value: float):
        self.total_value += value
        self.num_rollouts += 1
        if self.parent:
            # Negative, since parent is the other player
            self.parent.backpropagate(-value)

    def average_value(self):
        if self.num_rollouts == 0:
            return 0.0
        return self.total_value / self.num_rollouts

    def is_terminal(self):
        return self.game_state.is_over()

    def expand_children(self, move_probs_dict: dict):
        for move, prob in move_probs_dict.items():
            new_game_state = self.game_state.apply_move(move)
            self.children.append(MCTSNode(new_game_state, prob, self, move))

    # Select the child with the highest PUCT (Polynomial Upper Confidence for Trees) score.
    def select_child(self, c_puct: float):
        """
        Q(s,a) = child.average_value() (from the perspective of the current player 's')
        U(s,a) = c_puct * P(s,a) * (sqrt(N(s)) / (1 + N(s,a)))

        Our Q(s,a) is stored as child.average_value() and is from the perspective of the
        *parent* node. The current player wants to maximize Q(s,a) for the *next* state,
        which is equivalent to *minimizing* the Q value of the child node (which is from
        the opponent's perspective).

        Therefore, we use Q = -child.average_value()
        """
        total_parent_rollouts = self.num_rollouts

        best_score = -float('inf')
        best_child = None
        for child in self.children:
            # Q(s,a) = -child.average_value()
            exploitation_score = -child.average_value()
            # U(s,a)
            exploration_score = c_puct * child.prob * (math.sqrt(total_parent_rollouts) / (1 + child.num_rollouts))

            uct_score = exploitation_score + exploration_score

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child


class MCTSAgent(agent.Agent):
    def __init__(self, ac_agent: ACAgent, num_rounds: int, c_puct: float = 1.0, temperature: float = 1.0, lambda_mix: float = 0.0, device: str= "cpu"):
        agent.Agent.__init__(self)
        self.ac_agent = ac_agent

        self.num_rounds = num_rounds
        self.c_puct = c_puct
        self.temperature = temperature
        self.lambda_mix = lambda_mix
        self.ac_agent._model.eval()
        # Just for train() method
        self.device = device

    def select_moves(self, game_states: List[GameStateTemplate]) -> List[Move]:
        moves = []
        for game_state in game_states:
            # run_search returns the best policy and the selected move
            policy_target, selected_move = self.run_search(game_state)

            if selected_move is None:
                # This can happen if the game is over or no legal moves.
                # Fallback to a random move just in case.
                legal_moves = game_state.legal_moves()
                selected_move = random.choice(legal_moves) if legal_moves else None

            moves.append(selected_move)
        return moves

    def run_search(self, game_state: GameStateTemplate):
        root = MCTSNode(game_state, prob=1.0)
        for _ in range(self.num_rounds):
            node = root
            path = [root]

            # Get leaf node
            while node.children:
                assert node is not None, "'node' variable should not be be None"  # Can't happen, but to hide mypy warning
                node = node.select_child(self.c_puct)
            assert node is not None, "'node' variable should not be be None"  # Can't happen, but to hide mypy warning

            value = 0.0
            if node.is_terminal():
                winner = node.game_state.winner()
                if winner is None:
                    value = 0.0
                elif winner == node.game_state.next_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                with torch.no_grad():
                    move_probs, value = self.ac_agent.predict_policy_and_value([node.game_state])
                predicted_value = value[0].item()

                # Convert policy network's probs to dict used in node.expand_children()
                flattened_probs = move_probs[0]
                move_probs_dict = {}
                for move in node.game_state.legal_moves():
                    move_idx = self.ac_agent._encoder.encode_point(move.point)
                    prob = flattened_probs[move_idx].item()
                    move_probs_dict[move] = prob

                node.expand_children(move_probs_dict)

                if self.lambda_mix > 0:
                    rollout_reward = self.simulate_random_game(node.game_state)
                    value = (1-self.lambda_mix) * predicted_value + self.lambda_mix * rollout_reward
                else:
                    value = predicted_value  # Like in AlphaZero

            node.backpropagate(value)

            # Search is now completed
            num_moves: int = self.ac_agent._encoder.num_points()
            policy_target = torch.zeros(num_moves, dtype=torch.float32)
            move_to_child = {child.move: child for child in root.children}

            total_visits = root.num_rollouts - 1
            if total_visits <= 0:
                legal_moves: List[Move] = root.game_state.legal_moves()
                if not legal_moves:
                    return policy_target, None

                prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    move_idx = self.ac_agent._encoder.encode_point(move.point)
                    policy_target[move_idx] = prob

                selected_move = random.choice(legal_moves)
                return policy_target, selected_move

            child_visits = []
            child_moves = []
            for move in root.game_state.legal_moves():
                if move in move_to_child:
                    child = move_to_child[move]
                    move_idx = self.ac_agent._encoder.encode_point(move.point)
                    visits = child.num_rollouts
                    policy_target[move_idx] = visits / total_visits
                    child_visits.append(visits)
                    child_moves.append(move)

            # Select the move to play
            if self.temperature == 0:
                best_move_idx = torch.argmax(policy_target).item()
                selected_move = self.ac_agent._encoder.decode_point_index(best_move_idx)
            else:
                visits_with_temp = torch.tensor([v**(1.0 / self.temperature) for v in child_visits])
                draw_probs = visits_with_temp / torch.sum(visits_with_temp)
                move_idx = random.choices(range(len(child_moves)), weights=draw_probs, k=1)[0]
                selected_move = child_moves[move_idx]

            return policy_target, selected_move

    @staticmethod
    def simulate_random_game(game_state: GameStateTemplate):
        current_game = game_state.copy()

        if current_game.is_over():
            winner = current_game.winner()
        else:
            while True:
                legal_moves = current_game.legal_moves()
                if not legal_moves:
                    winner = None
                    break

                move = random.choice(legal_moves)
                current_game = current_game.apply_move(move)

                if current_game.is_over():
                    winner = current_game.winner()
                    break

        if winner is None:
            return 0.0
        elif winner == game_state.next_player:
            return 1.0
        else:
            return -1.0

    def train(self, experience: MCTSExperienceBuffer, lr:float=0.0001, batch_size:int=128, entropy_coef: float = 0.001, ppo_epochs: int = 3, clip_epsilon: float = 0.2):
        self.ac_agent._model.train()

        if self.optimizer is None:
            self.optimizer = Adam(self.ac_agent._model.parameters(), lr=lr)
        value_loss_fn = nn.MSELoss()

        dataset = TensorDataset(
            experience.states,
            experience.policy_targets,
            experience.value_targets,
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0
        total_value_loss = 0
        total_combined_loss = 0

        total_grad_norm_before = 0
        total_grad_norm_after = 0
        num_batches = 0

        for _ in tqdm(range(ppo_epochs), desc="Training agent"):
            for states_batch, policy_targets_batch, winner_targets_batch in data_loader:
                states_batch = states_batch.to(self.device)
                policy_targets_batch = policy_targets_batch.to(self.device)
                winner_targets_batch = winner_targets_batch.to(self.device)

                self.optimizer.zero_grad()
                policy_logits, value_pred = self.ac_agent._model(states_batch)

                # TODO: Verify masking works properly!
                # Mask illegal moves
                # is_game_over = ~masks_batch.any(dim=1)
                # if is_game_over.any():
                #     masks_batch[is_game_over, 0] = True
                # masked_logits = policy_logits.clone()
                # masked_logits[~masks_batch] = float('-inf')
                # dist = torch.distributions.Categorical(logits=masked_logits)
                # new_log_probs = dist.log_prob(actions_batch)

                # TODO: PPO loss function!
                # ratio = torch.exp(new_log_probs - old_log_probs_batch)
                # surr1 = ratio * advantages_batch
                # surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch
                # policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss = -torch.sum(policy_targets_batch * F.log_softmax(policy_logits, dim=1), dim=1).mean()

                value_loss = F.mse_loss(value_pred, winner_targets_batch)

                # entropy_loss = -entropy_coef * dist.entropy().mean()
                entropy_loss = 0.0

                total_loss = policy_loss + 0.5 * value_loss + entropy_loss

                total_loss.backward()
                grad_norm_before = torch.nn.utils.clip_grad_norm_(self.ac_agent._model.parameters(), max_norm=0.5)
                grad_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in self.ac_agent._model.parameters() if p.grad is not None)) # pyright: ignore[reportArgumentType]
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_combined_loss += total_loss.item()

                total_grad_norm_before += grad_norm_before
                total_grad_norm_after += grad_norm_after
                num_batches += 1

        return (
            total_policy_loss / num_batches,
            total_value_loss / num_batches,
            total_combined_loss / num_batches,
            total_grad_norm_before / num_batches,
            total_grad_norm_after / num_batches,
        )
