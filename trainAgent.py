import argparse

from DLCF import rl
from Model import Model
from typing import List

def trainAgent(learning_agent_filename: str, experience_files: List[str], updated_agent_filename: str, learning_rate: float, batch_size: int, device: str = "cpu"):
    learning_agent = rl.ACAgent.load(learning_agent_filename, Model, device=device)

    total_policy_loss = 0
    total_value_loss = 0
    total_combined_loss = 0
    num_exp_files = len(experience_files)

    for exp_filename in experience_files:
        exp_buffer = rl.ExperienceBuffer.load(exp_filename)

        policy_loss, value_loss, combined_loss = learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size)

        total_policy_loss += policy_loss
        total_value_loss += value_loss
        total_combined_loss += combined_loss
        num_exp_files +=1

    learning_agent.save(updated_agent_filename)

    return (total_policy_loss / num_exp_files,
            total_value_loss / num_exp_files,
            total_combined_loss / num_exp_files)

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
