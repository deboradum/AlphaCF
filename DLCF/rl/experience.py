import torch
from typing import List

class ExperienceCollector:
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.advantages: List[torch.Tensor] = []
        self.old_log_probs: List[torch.Tensor] = []

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

        if num_states == 0:
            return

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

        self.states.append(torch.stack(self._current_episode_states))
        self.actions.append(torch.tensor(self._current_episode_actions, dtype=torch.long))
        self.old_log_probs.append(torch.tensor(self._current_episode_log_probs, dtype=torch.float32))
        self.advantages.append(torch.tensor(advantages, dtype=torch.float32))
        self.rewards.append(torch.tensor(value_targets, dtype=torch.float32))

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_episode_log_probs = []

    def to_buffer(self):
        if not self.states:
            raise ValueError("No experience collected to create a buffer.")

        return ExperienceBuffer(
            states=torch.cat(self.states, dim=0),
            actions=torch.cat(self.actions, dim=0),
            rewards=torch.cat(self.rewards, dim=0),
            advantages=torch.cat(self.advantages, dim=0),
            old_log_probs=torch.cat(self.old_log_probs, dim=0)
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
        data = torch.load(path, weights_only=True)
        return cls(
            states=data['states'],
            actions=data['actions'],
            rewards=data['rewards'],
            advantages=data['advantages'],
            old_log_probs=data['old_log_probs'],
        )


# collectors_lists is [ [c1_game1, c1_game2, ...], [c2_game1, c2_game2, ...] ]
def combine_experience(collectors_lists: List[List[ExperienceCollector]]):
    all_collectors = [c for c_list in collectors_lists for c in c_list]

    all_states = [tensor for c in all_collectors for tensor in c.states]
    all_actions = [tensor for c in all_collectors for tensor in c.actions]
    all_rewards = [tensor for c in all_collectors for tensor in c.rewards]
    all_advantages = [tensor for c in all_collectors for tensor in c.advantages]
    all_old_log_probs = [tensor for c in all_collectors for tensor in c.old_log_probs]

    if not all_states:
        raise ValueError("No experience collected to combine.")

    return ExperienceBuffer(
        states=torch.cat(all_states, dim=0),
        actions=torch.cat(all_actions, dim=0),
        rewards=torch.cat(all_rewards, dim=0),
        advantages=torch.cat(all_advantages, dim=0),
        old_log_probs=torch.cat(all_old_log_probs, dim=0)
    )
