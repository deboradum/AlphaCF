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
    def __init__(
        self,
        ac_agent: ACAgent,
        num_rounds: int,
        c_puct: float = 1.0,
        temperature: float = 0.0,
        lambda_mix: float = 0.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        eval_mode: bool = False,
    ):
        agent.Agent.__init__(self)
        self.ac_agent = ac_agent

        self.num_rounds = num_rounds
        self.c_puct = c_puct
        self.temperature = temperature
        self.lambda_mix = lambda_mix
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        if eval_mode:
            self.ac_agent._model.eval()

    def select_moves(self, game_states: List[GameStateTemplate]) -> List[Move]:
        bs = len(game_states)

        moves = []
        for game_state in game_states:
            (
                Xs,
                action_tensors,
                _,
                _,
                _,
                valid_move_mask,
                mcts_policy_target,
                mcts_value_target,
                selected_move
            ) = self._run_search(game_state)

            if self.ac_agent._collectors is not None and Xs is not None:
                action_indices = action_tensors.tolist()
                mcts_policy_target_list = mcts_policy_target.tolist()
                # mcts_value_target_list = mcts_value_target.tolist()
                for i in range(bs):
                    self.ac_agent._collectors[i].record_decision(
                        state=Xs[i],
                        action=action_indices[i],
                        log_prob=mcts_policy_target_list[i],
                        estimated_value=mcts_value_target,  # TODO: change this if I want batching
                        mask=valid_move_mask[i]
                    )

            # Should never happen
            if selected_move is None:
                legal_moves = game_state.legal_moves()
                selected_move = random.choice(legal_moves) if legal_moves else None

            moves.append(selected_move)
        return moves

    def _run_search(self, game_state: GameStateTemplate):
        root = MCTSNode(game_state, prob=1.0)

        root_data = {
            "Xs": None,
            "action_tensors": None,
            "estimated_values": None,
            "log_probs": None,
            "move_probs": None,
            "valid_move_mask": None
        }
        # Tree search
        for i in range(self.num_rounds):
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
                    Xs, action_tensors, estimated_values, log_probs, move_probs, valid_move_mask = self.ac_agent.sample_moves([node.game_state])
                if i == 0:
                    root_data["Xs"] = Xs
                    root_data["action_tensors"] = action_tensors
                    root_data["estimated_values"] = estimated_values
                    root_data["log_probs"] = log_probs
                    root_data["move_probs"] = move_probs
                    root_data["valid_move_mask"] = valid_move_mask

                predicted_value = estimated_values[0].item()

                # Convert policy network's probs to dict used in node.expand_children()
                flattened_probs = move_probs[0]
                move_probs_dict = {}

                legal_moves_list = node.game_state.legal_moves()
                for move in legal_moves_list:
                    move_idx = self.ac_agent._encoder.encode_point(move.point)
                    prob = flattened_probs[move_idx].item()
                    move_probs_dict[move] = prob
                # Dirichlet noise
                if node is root and self.dirichlet_epsilon > 0:
                    num_legal_moves = len(legal_moves_list)
                    if num_legal_moves > 0:
                        alphas = torch.full((num_legal_moves,), self.dirichlet_alpha)
                        dir_dist = torch.distributions.dirichlet.Dirichlet(alphas)
                        noise = dir_dist.sample()

                        # Mix noise into probs
                        for i, move in enumerate(legal_moves_list):
                            original_prob = move_probs_dict[move]
                            noise_prob = noise[i].item()
                            move_probs_dict[move] = (1 - self.dirichlet_epsilon) * original_prob + self.dirichlet_epsilon * noise_prob

                node.expand_children(move_probs_dict)

                if self.lambda_mix > 0:
                    rollout_reward = self.simulate_random_game(node.game_state)
                    value = (1-self.lambda_mix) * predicted_value + self.lambda_mix * rollout_reward
                else:
                    value = predicted_value  # Like in AlphaZero

            node.backpropagate(value)

        # Search is now completed
        num_moves: int = self.ac_agent._encoder.num_points()
        policy_target = torch.zeros(num_moves, dtype=torch.float32) # This is the MCTS policy target
        move_to_child = {child.move: child for child in root.children}

        total_visits = root.num_rollouts - 1
        if total_visits <= 0:
            legal_moves: List[Move] = root.game_state.legal_moves()
            if not legal_moves:
                raise Exception("No legal moves. This should never happen I think!")
                print("No legal moves!") # TODO: Should never happen I think?
                selected_move = None
                mcts_value = 0.0 # No rollouts, value is 0
                return (
                    root_data["Xs"], root_data["action_tensors"], root_data["estimated_values"],
                    root_data["log_probs"], root_data["move_probs"], root_data["valid_move_mask"],
                    policy_target, mcts_value, selected_move
                )

            prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_idx = self.ac_agent._encoder.encode_point(move.point)
                policy_target[move_idx] = prob

            selected_move = random.choice(legal_moves)
            mcts_value = root.average_value()
            return (
                root_data["Xs"], root_data["action_tensors"], root_data["estimated_values"],
                root_data["log_probs"], root_data["move_probs"], root_data["valid_move_mask"],
                policy_target, mcts_value, selected_move
            )

        child_visits = []
        child_moves = []
        for child in root.children:
            if child.num_rollouts > 0:
                move_idx = self.ac_agent._encoder.encode_point(child.move.point)
                visits = child.num_rollouts

                policy_target[move_idx] = visits / total_visits

                child_visits.append(visits)
                child_moves.append(child.move)

        # Select the move to play
        if self.temperature == 0:
            best_move_idx = torch.argmax(policy_target).item()
            selected_move = self.ac_agent._encoder.decode_point_index(best_move_idx)
        else:
            visits_with_temp = torch.tensor([v**(1.0 / self.temperature) for v in child_visits])
            draw_probs = visits_with_temp / torch.sum(visits_with_temp)
            move_idx = random.choices(range(len(child_moves)), weights=draw_probs, k=1)[0]
            selected_move = child_moves[move_idx]

        mcts_value = root.average_value()

        return (
            root_data["Xs"],
            root_data["action_tensors"],
            root_data["estimated_values"],
            root_data["log_probs"],
            root_data["move_probs"],
            root_data["valid_move_mask"],
            policy_target,
            mcts_value,
            selected_move
        )

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
