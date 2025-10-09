import torch
from typing import List

class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state: torch.Tensor, action: int, estimated_value:float=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward: float):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - \
                self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def to_buffer(self):
        return ExperienceBuffer(
            states=torch.Tensor(self.states),
            actions=torch.Tensor(self.actions),
            rewards=torch.Tensor(self.rewards),
            advantages=torch.Tensor(self.advantages))

class ExperienceBuffer:
    def __init__(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def save(self, path: str):
        torch.save({
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'advantages': self.advantages
        }, path)

    @classmethod
    def load(cls, path: str):
        data = torch.load(path)
        return cls(
            states=data['states'],
            actions=data['actions'],
            rewards=data['rewards'],
            advantages=data['advantages']
        )

def combine_experience(collectors: List[ExperienceCollector]):
    combined_states = torch.cat([torch.stack(c.states) for c in collectors], dim=0)
    combined_actions = torch.cat([torch.tensor(c.actions, dtype=torch.long) for c in collectors], dim=0)
    combined_rewards = torch.cat([torch.tensor(c.rewards, dtype=torch.float32) for c in collectors], dim=0)
    combined_advantages = torch.cat([torch.tensor(c.advantages, dtype=torch.float32) for c in collectors], dim=0)

    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards,
        combined_advantages)
