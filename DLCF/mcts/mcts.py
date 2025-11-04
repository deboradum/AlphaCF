import math
import random

from DLCF import agent
from DLCF.rl import ACAgent
from DLCF.connectFourBoard import GameState
from DLCF.getGameState import Player, GameStateTemplate, Move

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

        self.children = []
        self.prob = prob

    def add_random_child(self, ac_agent: ACAgent):
        move_probs, Xs, action_tensors, estimated_values, log_probs = ac_agent.sample_moves(self.game_state, return_probs=True)
        move_probs.squeeze_(0)

        move_probs_list = move_probs.tolist()

        # TODO: Verify this is correct.
        new_node = None
        highest_prob = -1
        point_indices = action_tensors.tolist()
        for i, idx in enumerate(point_indices):
            new_move = self.univisted_moves.pop(idx)
            new_game_state = self.game_state.apply_move(new_move)
            prob = move_probs_list[i]
            curr_child_node = MCTSNode(new_game_state, prob, self, new_move)
            self.children.append(new_node)
            # Return node with highest action prob.
            if prob > highest_prob:
                highest_prob = prob
                new_node = curr_child_node

        return new_node

    def backpropagate(self, value: float):
        self.total_value += value
        self.num_rollouts += 1

    def average_value(self):
        if self.num_rollouts == 0:
            return 0.0
        return self.total_value / self.num_rollouts

    def is_terminal(self):
        return self.game_state.is_over()

    def expand_children(self, move_probs):
        for move in self.game_state.legal_moves():
            prob = move_probs.get(move, 0.0)
            new_game_state = self.game_state.apply_move(move)
            self.children.append(MCTSNode(new_game_state, prob, self, move))


class MCTSAgent(agent.Agent):
    def __init__(self, ac_agent: ACAgent, num_rounds: int, temperature: float, lambda_: float):
        agent.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.lambda_ = lambda_
        self.ac_agent = ac_agent  # TODO: Need to set require gradient to false maybe?
        self.ac_agent.eval()

    def select_move(self, game_state: GameState):
        root = MCTSNode(game_state, prob=1.0)
        path = [root]

        for i in range(self.num_rounds):
            # Selection
            while node.children:
                node = self.select_child(node)
                path.append(node)
            # 'node' is now a leaf node

            # Expansion + evaluation
            if node.is_terminal():
                # Terminal state, get true value
                winner = node.game_state.winner()
                current_player_at_node = node.game_state.next_player
                value = 0.0
                if winner is not None:
                    # Value is from the perspective of the current player
                    if winner == current_player_at_node:
                        value = 1.0
                    else:
                        value = -1.0
            else:
                move_probs, v = self.ac_agent.predict_policy_and_value(node.game_state)
                node.expand_children(move_probs)

                value_nn = v.item()
                winner = self.simulate_random_game(node.game_state)
                rollout_reward = 0.0
                if winner == node.game_state.next_player:
                    rollout_reward = 1.0
                elif winner is not None: # i.e., winner is the other player
                    rollout_reward = -1.0
                value = (1-self.lambda_) * value_nn + self.lambda_ * rollout_reward

            # Backpropegate
            for node_in_path in reversed(path):
                node_in_path.backpropagate(value)
                value = -value # Invert value for the parent

        print("MCTS Search Results:")
        scored_moves = [
            (child.average_value(), child.move, child.num_rollouts)
            for child in root.children
        ]
        # Sort by visit count
        scored_moves.sort(key=lambda x: x[2], reverse=True)

        for v, m, n in scored_moves[:10]:
            print('%s - Q: %.3f (visits: %d)' % (m, v, n)) # Print avg_value (Q)

        if not root.children:
            return None # No legal moves

        # Select move with the most simulations
        best_child = max(root.children, key=lambda child: child.num_rollouts)
        best_move = best_child.move

        print('Select move %s with %d visits' % (best_move, best_child.num_rollouts))

        return best_move

    def select_child(self, node: MCTSNode):
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score = -float('inf')
        best_child = None
        for child in node.children:
            win_percentage = -1.0 * child.average_value()

            exploration_factor = child.prob*(math.sqrt(total_rollouts) / (1+child.num_rollouts))

            uct_score = win_percentage + self.temperature * exploration_factor

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    @staticmethod
    def simulate_random_game(game: GameState):
        bots = {
            Player.black: agent.RandomBot(),
            Player.white: agent.RandomBot()
        }
        current_game = game.clone()

        while not game.is_over():
            bot_move = bots[current_game.next_player].select_move(current_game)
            current_game = current_game.apply_move(bot_move)

        return current_game.winner()
