import torch
from typing import List

class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.old_log_probs = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_episode_log_probs = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_episode_log_probs = []

    def record_decision(self, state: torch.Tensor, action: int, log_prob: float, estimated_value:float=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)
        self._current_episode_log_probs.append(log_prob)

    def complete_episode(self, reward: float, gamma: float = 0.99, lambda_: float = 0.95):
        num_states = len(self._current_episode_states)

        # GAE
        advantages = [0.0] * num_states
        value_targets = [0.0] * num_states
        gae = 0.0
        next_value = float(reward)
        for i in reversed(range(num_states)):
            reward_at_step = 0 if i < num_states - 1 else reward
            current_value = self._current_episode_estimated_values[i]
            value_targets[i] = reward_at_step + gamma * next_value

            # δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = reward_at_step + gamma * next_value - current_value
            gae = delta + gamma * lambda_ * gae
            advantages[i] = gae

            next_value = current_value

        self.states.extend(self._current_episode_states)
        self.actions.extend(self._current_episode_actions)
        self.old_log_probs.extend(self._current_episode_log_probs)
        self.advantages.extend(advantages)
        self.rewards.extend(value_targets)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_episode_log_probs = []

    def to_buffer(self):
        return ExperienceBuffer(
            states=torch.stack(self.states),
            actions=torch.Tensor(self.actions, dtype=torch.long),
            rewards=torch.Tensor(self.rewards, dtype=torch.float32),
            advantages=torch.Tensor(self.advantages, dtype=torch.float32),
            old_log_probs=torch.tensor(self.old_log_probs, dtype=torch.float32)
        )


class ExperienceBuffer:
    def __init__(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor, old_log_probs: torch.Tensor):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages
        self.old_log_probs = old_log_probs

    def save(self, path: str):
        torch.save({
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'advantages': self.advantages,
            'old_log_probs': self.old_log_probs,
        }, path)

    @classmethod
    def load(cls, path: str):
        data = torch.load(path)
        return cls(
            states=data['states'],
            actions=data['actions'],
            rewards=data['rewards'],
            advantages=data['advantages'],
            old_log_probs=data['old_log_probs'],
        )

def combine_experience(collectors: List[ExperienceCollector]):
    combined_states = [state for c in collectors for state in c.states]
    combined_actions = [action for c in collectors for action in c.actions]
    combined_rewards = [reward for c in collectors for reward in c.rewards]
    combined_advantages = [advantage for c in collectors for advantage in c.advantages]
    combined_old_log_probs = [log_prob for c in collectors for log_prob in c.old_log_probs]

    return ExperienceBuffer(
        states=torch.stack(combined_states),
        actions=torch.tensor(combined_actions, dtype=torch.long),
        rewards=torch.tensor(combined_rewards, dtype=torch.float32),
        advantages=torch.tensor(combined_advantages, dtype=torch.float32),
        old_log_probs=torch.tensor(combined_old_log_probs, dtype=torch.float32)
    )
