import h5py
import argparse

from DLCF import rl
from Model import Model
from typing import List

def trainAgent(learning_agent_filename: str, experience_files: List[str], updated_agent_filename: str, learning_rate: float, batch_size: int, device: str = "cpu"):
    learning_agent = rl.load_ac_agent(h5py.File(learning_agent_filename), Model, device=device)

    for exp_filename in experience_files:
        exp_buffer = rl.load_experience(h5py.File(exp_filename))

        learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size)

    with h5py.File(updated_agent_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', type=str, required=True)
    parser.add_argument('--agent-out', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to run on (cpu, cuda, or mps)')
    parser.add_argument('experience', nargs='+')
    args = parser.parse_args()

    learning_agent_filename = args.learning_agent
    experience_files = args.experience
    updated_agent_filename = args.agent_out
    learning_rate = args.lr
    batch_size = args.bs
    device = args.device

    trainAgent(learning_agent_filename, experience_files, updated_agent_filename, learning_rate, batch_size, device=device)
