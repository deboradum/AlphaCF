import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from Model import Model
from DLCF.rl import ACAgent
from DLCF.rl.mctsExperience import MCTSExperienceBuffer

def mctsTrainAgent(
    learning_agent_filename: str,
    experience_buffer: MCTSExperienceBuffer,
    updated_agent_filename: str,
    learning_rate: float,
    batch_size: int,
    train_epochs: int,
    l2_reg: float,
    device: str = "cpu"
):
    # Load the agent you want to train
    learning_agent = ACAgent.load(learning_agent_filename, Model, device=device)
    learning_agent._model.train() # Set to train mode

    optimizer = Adam(learning_agent._model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    total_policy_loss_epoch = 0
    total_value_loss_epoch = 0
    total_combined_loss_epoch = 0

    num_batches = len(experience_buffer) // batch_size
    if num_batches == 0 and len(experience_buffer) > 0:
        num_batches = 1
        print(f"Warning: Batch size {batch_size} is larger than buffer size {len(experience_buffer)}. Running one batch.")
    elif num_batches == 0:
        print("Error: Experience buffer is empty. Cannot train.")
        return 0, 0, 0

    for _ in tqdm(range(train_epochs), desc="Training agent on MCTS data"):
        total_policy_loss = 0
        total_value_loss = 0
        total_combined_loss = 0

        data_loader = DataLoader(experience_buffer.buffer, batch_size=batch_size, shuffle=True)

        for states_batch, policy_targets_batch, winner_targets_batch in data_loader:
            states_batch = states_batch.to(device)
            policy_targets_batch = policy_targets_batch.to(device)
            winner_targets_batch = winner_targets_batch.to(device).view(-1, 1) # (B, 1)

            optimizer.zero_grad()

            policy_logits, value_pred = learning_agent._model(states_batch)

            # Value Loss
            value_loss = F.mse_loss(value_pred, winner_targets_batch)

            # Policy Loss
            policy_loss = F.cross_entropy(policy_logits, policy_targets_batch)

            total_loss = policy_loss + value_loss

            total_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_combined_loss += total_loss.item()

        total_policy_loss_epoch += (total_policy_loss / num_batches)
        total_value_loss_epoch += (total_value_loss / num_batches)
        total_combined_loss_epoch += (total_combined_loss / num_batches)

    learning_agent.save(updated_agent_filename)

    avg_policy_loss = total_policy_loss_epoch / train_epochs
    avg_value_loss = total_value_loss_epoch / train_epochs
    avg_combined_loss = total_combined_loss_epoch / train_epochs

    print(f"Training complete. Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

    return avg_policy_loss, avg_value_loss, avg_combined_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', type=str, required=True, help="Agent to load and train.")
    parser.add_argument('--agent-out', type=str, required=True, help="Path to save the new agent.")
    parser.add_argument('--experience-in', type=str, required=True, help="Path to the MCTSExperienceBuffer file.")
    parser.add_argument('--buffer-size', type=int, default=20000, help="Max size of the experience buffer (must match buffer creation).")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--train-epochs', type=int, default=3)
    parser.add_argument('--l2-reg', type=float, default=1e-4, help="L2 Regularization strength (weight decay).")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu')

    args = parser.parse_args()

    try:
        experience_buffer = MCTSExperienceBuffer.load(args.experience_in, args.buffer_size)
        print(f"Loaded {len(experience_buffer)} samples from {args.experience_in}")
    except FileNotFoundError:
        print(f"Could not find experience file: {args.experience_in}")
        exit(1)
    except Exception as e:
        print(f"Error loading buffer: {e}")
        exit(1)

    mctsTrainAgent(
        learning_agent_filename=args.learning_agent,
        experience_buffer=experience_buffer,
        updated_agent_filename=args.agent_out,
        learning_rate=args.lr,
        batch_size=args.bs,
        train_epochs=args.train_epochs,
        l2_reg=args.l2_reg,
        device=args.device
    )
