import argparse

from DLCF import rl
from typing import List
from Model import Model
from DLCF.rl import ACAgent
from DLCF.mcts import MCTSAgent

def mctsTrainAgent(
    learning_agent_filename: str,
    experience_files: List[str],
    updated_agent_filename: str,
    learning_rate: float,
    batch_size: int,
    entropy_coef: float,
    ppo_epochs: int,
    clip_epsilon: float,
    device: str = "cpu"
):
    ac_agent = ACAgent.load(learning_agent_filename, Model, device=device)
    learning_agent = MCTSAgent(
        ac_agent=ac_agent,
        num_rounds=100,
    )

    total_policy_loss = 0
    total_entropy_loss = 0
    total_value_loss = 0
    total_combined_loss = 0
    total_grad_norm_before = 0
    total_grad_norm_after = 0
    num_exp_files = len(experience_files)

    for exp_filename in experience_files:
        exp_buffer = rl.ExperienceBuffer.load(exp_filename)

        policy_loss, entropy_loss, value_loss, combined_loss, grad_norm_before, grad_norm_after = learning_agent.ac_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size,
            entropy_coef=entropy_coef,
            ppo_epochs=ppo_epochs,
            clip_epsilon=clip_epsilon,
        )

        total_policy_loss += policy_loss
        total_entropy_loss += entropy_loss
        total_value_loss += value_loss
        total_combined_loss += combined_loss
        total_grad_norm_before += grad_norm_before
        total_grad_norm_after += grad_norm_after

    learning_agent.ac_agent.save(updated_agent_filename)

    return (
        total_policy_loss / num_exp_files,
        total_entropy_loss / num_exp_files,
        total_value_loss / num_exp_files,
        total_combined_loss / num_exp_files,
        total_grad_norm_before / num_exp_files,
        total_grad_norm_after / num_exp_files,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', type=str, required=True, help="Agent to load and train.")
    parser.add_argument('--agent-out', type=str, required=True, help="Path to save the new agent.")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--ppo-epochs', type=int, default=3)
    parser.add_argument('--clip-epsilon', type=float, default=0.02)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()

    learning_agent_filename = args.learning_agent
    experience_files = args.experience
    updated_agent_filename = args.agent_out
    learning_rate = args.lr
    batch_size = args.bs
    entropy_coef = args.entropy_coef
    ppo_epochs = args.ppo_epochs
    clip_epsilon = args.clip_epsilon
    device = args.device

    mctsTrainAgent(
        learning_agent_filename=learning_agent_filename,
        experience_files=experience_files,
        updated_agent_filename=updated_agent_filename,
        learning_rate=learning_rate,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        device=device
    )
