import torch
from typing import List

class MCTSExperienceCollector:
    def __init__(self):
        self.states = []
        self.policy_targets = []

        self.winner = 0.0

    def begin_episode(self):
        self.states = []
        self.policy_targets = []
        self.winner = 0.0

    def record_decision(self, state: torch.Tensor, policy_target: torch.Tensor):
        self.states.append(state)
        self.policy_targets.append(policy_target)

    def complete_episode(self, winner_val: float):
        self.winner = winner_val

    def to_buffer_tuples(self):
        if not self.states:
            return []

        return [
            (state, policy, torch.tensor(self.winner, dtype=torch.float32))
            for state, policy in zip(self.states, self.policy_targets)
        ]


class MCTSExperienceBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = [] # [(state, policy, winner)]
        self.pos = 0

    def add_game(self, collector: MCTSExperienceCollector):
        game_data = collector.to_buffer_tuples()
        for experience in game_data:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.pos] = experience
                self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, batch_size: int):
        indices = torch.randint(low=0, high=len(self.buffer), size=(batch_size,))

        states, policies, winners = zip(*[self.buffer[i] for i in indices])

        return (
            torch.stack(states),
            torch.stack(policies),
            torch.stack(winners)
        )

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        torch.save(self.buffer, path)

    @classmethod
    def load(cls, path: str, buffer_size: int):
        buffer_data = torch.load(path)
        new_buffer = cls(buffer_size)
        new_buffer.buffer = buffer_data
        new_buffer.pos = len(buffer_data) % buffer_size
        return new_buffer
