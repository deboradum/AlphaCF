import math
import torch
import random
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from DLCF import agent
from DLCF.rl import ACAgent
from DLCF.DLCFtypes import Move
from DLCF.getGameState import GameStateTemplate

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
        self.prob = prob  # prior probability p(s,a)

        # Caching for vectorization
        self.child_priors = None
        self.child_moves = None

    def backpropagate(self, value: float):
        self.total_value += value
        self.num_rollouts += 1
        if self.parent:
            self.parent.backpropagate(-value)

    def average_value(self):
        if self.num_rollouts == 0:
            return 0.0
        return self.total_value / self.num_rollouts

    def is_terminal(self):
        return self.game_state.is_over()

    def expand_children(self, moves: List[Move], probs: np.ndarray):
        """
        Optimized expansion: receives a numpy array of probs directly.
        """
        self.children = []
        # Pre-allocate lists for vectorized selection later
        self.child_priors = np.zeros(len(moves), dtype=np.float32)
        self.child_moves = []

        for i, move in enumerate(moves):
            new_game_state = self.game_state.apply_move(move)
            prob = probs[i]
            child = MCTSNode(new_game_state, prob, self, move)
            self.children.append(child)

            # Cache for vectorized selection
            self.child_priors[i] = prob
            self.child_moves.append(move)

    def select_child_vectorized(self, c_puct: float):
        """
        Vectorized UCT calculation using NumPy.
        Huge speedup over iterating python objects.
        """
        # 1. Extract stats from children
        # Ideally, we would update these arrays incrementally, but rebuilding
        # valid arrays for this list size is still faster than a Python for-loop.
        counts = np.array([c.num_rollouts for c in self.children], dtype=np.float32)
        # Note: We want -avg_value because we want to minimize opponent's Q
        avg_values = np.array([-c.average_value() for c in self.children], dtype=np.float32)

        # 2. Calculate UCT
        sqrt_parent = math.sqrt(self.num_rollouts)

        # U = c_puct * P * (sqrt(N_parent) / (1 + N_child))
        u_scores = c_puct * self.child_priors * (sqrt_parent / (1 + counts))

        # Q + U
        uct_scores = avg_values + u_scores

        # 3. Argmax
        best_idx = np.argmax(uct_scores)
        return self.children[best_idx]


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

    def run_mcts(self, game_states: List[GameStateTemplate]) -> Tuple[List[torch.Tensor], List[float], List[MCTSNode], List[Dict]]:
        """
        Runs MCTS simulations and returns the resulting policy and value.

        Returns:
            - List[torch.Tensor]: Policy targets for each root state.
            - List[float]: Estimated values for each root state.
            - List[MCTSNode]: The root nodes.
            - List[Dict]: The root datasets for collectors.
        """
        batch_size = len(game_states)
        roots = [MCTSNode(gs, prob=1.0) for gs in game_states]
        root_datasets = [{} for _ in range(batch_size)]

        for i in range(self.num_rounds):
            leaf_nodes = []
            terminals = []

            for root in roots:
                node = root
                while node.children:
                    node = node.select_child_vectorized(self.c_puct)
                leaf_nodes.append(node)

            # Separate Terminal vs Non-Terminal
            expandable_pairs = [] # (index_in_batch, node)
            for idx, node in enumerate(leaf_nodes):
                if node.is_terminal():
                    # Handle terminal value immediately
                    winner = node.game_state.winner()
                    if winner is None: val = 0.0
                    elif winner == node.game_state.next_player: val = 1.0
                    else: val = -1.0
                    node.backpropagate(val)
                else:
                    expandable_pairs.append((idx, node))

            if not expandable_pairs:
                continue

            states_to_eval = [p[1].game_state for p in expandable_pairs]

            with torch.no_grad():
                Xs, _, estimated_values, _, move_probs, valid_move_mask = self.ac_agent.sample_moves(states_to_eval)
            move_probs_cpu = move_probs.cpu().numpy()

            for batch_idx, (original_idx, node) in enumerate(expandable_pairs):
                if i == 0:
                    root_datasets[original_idx] = {
                        "Xs": Xs[batch_idx],
                        "valid_move_mask": valid_move_mask[batch_idx],
                        "estimated_value": estimated_values[batch_idx]
                    }

                value = estimated_values[batch_idx].item()
                legal_moves = node.game_state.legal_moves()
                move_indices = [self.ac_agent._encoder.encode_point(m.point) for m in legal_moves]
                node_probs = move_probs_cpu[batch_idx, move_indices]

                prob_sum = np.sum(node_probs)
                if prob_sum > 0:
                    node_probs /= prob_sum

                # Dirichlet Noise (only at root)
                if node.parent is None and self.dirichlet_epsilon > 0 and len(legal_moves) > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
                    node_probs = (1 - self.dirichlet_epsilon) * node_probs + self.dirichlet_epsilon * noise

                node.expand_children(legal_moves, node_probs)

                if self.lambda_mix > 0:
                    rollout_reward = self.simulate_random_game(node.game_state)
                    value = (1 - self.lambda_mix) * value + self.lambda_mix * rollout_reward

                node.backpropagate(value)

        # --- End of Simulation Loop ---

        # Calculate final policies and values
        all_policy_targets = []
        all_root_values = []
        num_encoder_moves = self.ac_agent._encoder.num_points()

        for root in roots:
            all_root_values.append(root.average_value())

            if not root.children:
                all_policy_targets.append(torch.zeros(num_encoder_moves, dtype=torch.float32))
                continue

            total_visits = root.num_rollouts - 1
            policy_target_list = [0.0] * num_encoder_moves
            child_moves = root.children

            for child in child_moves:
                if child.num_rollouts > 0:
                    visits = child.num_rollouts
                    idx = self.ac_agent._encoder.encode_point(child.move.point)
                    if total_visits > 0:
                        policy_target_list[idx] = visits / total_visits

            all_policy_targets.append(torch.tensor(policy_target_list, dtype=torch.float32))

        return all_policy_targets, all_root_values, roots, root_datasets


    def select_moves(self, game_states: List[GameStateTemplate]) -> List[Move]:
        # Run the MCTS simulation to get policies, values, and roots
        all_policy_targets, _, roots, root_datasets = self.run_mcts(game_states)

        selected_moves = []
        for i, root in enumerate(roots):
            if not root.children:
                selected_moves.append(None)
                continue

            policy_target_tensor = all_policy_targets[i] # This is a tensor
            child_moves = root.children # Access the objects

            # Gather visit counts for move selection
            child_visits = []
            valid_children = []
            for child in child_moves:
                if child.num_rollouts > 0:
                    child_visits.append(child.num_rollouts)
                    valid_children.append(child)

            # Select Move
            if not valid_children:
                selected_move = random.choice(root.game_state.legal_moves())
            elif self.temperature == 0:
                max_visit = -1
                best_c = None
                for c in valid_children:
                    if c.num_rollouts > max_visit:
                        max_visit = c.num_rollouts
                        best_c = c
                selected_move = best_c.move
            else:
                visits_arr = np.array(child_visits, dtype=np.float64)
                visits_temp = visits_arr ** (1.0 / self.temperature)
                sum_v = np.sum(visits_temp)
                if sum_v > 0:
                    probs = visits_temp / sum_v
                    chosen_child = np.random.choice(valid_children, p=probs)
                    selected_move = chosen_child.move
                else:
                    selected_move = valid_children[0].move

            if self.ac_agent._collectors is not None and "Xs" in root_datasets[i]:
                rd = root_datasets[i]
                action_idx = self.ac_agent._encoder.encode_point(selected_move.point)
                move_prob = policy_target_tensor[action_idx].item()
                log_prob_scalar = torch.log(torch.tensor(move_prob + 1e-10)).item()
                self.ac_agent._collectors[i].record_decision(
                    state=rd["Xs"],
                    action=action_idx,
                    log_prob=log_prob_scalar,
                    policy_target=policy_target_tensor, # Pass the full policy tensor
                    estimated_value=rd["estimated_value"],
                    mask=rd["valid_move_mask"],
                )

            selected_moves.append(selected_move)

        return selected_moves

    @staticmethod
    def simulate_random_game(game_state: GameStateTemplate):
        current_game = game_state.copy()
        while not current_game.is_over():
            legal_moves = current_game.legal_moves()
            if not legal_moves: break
            move = random.choice(legal_moves)
            current_game = current_game.apply_move(move)

        winner = current_game.winner()
        if winner is None: return 0.0
        return 1.0 if winner == game_state.next_player else -1.0
