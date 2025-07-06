'''
Author: David Megli
Date: 2025-05-02
File: dqn.py
Description: Main Deep Q-Network Algorithm (training loop)
'''
import torch
import torch.nn.functional as F
import random
import numpy as np

from common import evaluate_policy, save_checkpoint
from replay_buffer import ReplayBuffer


def train_dqn(env, q_net, target_net, optimizer,
              episodes=500,
              gamma=0.99,
              batch_size=64,
              buffer_capacity=100000,
              min_buffer_size=1000,
              epsilon_start=1.0,
              epsilon_end=0.05,
              epsilon_decay=0.995,
              target_update=100,
              eval_every=50,
              run=None,
              checkpoint_dir=None,
              device=torch.device("cpu")):
    """
    Train a Deep Q-Network (DQN) on the given environment.
    
    This function implements the DQN algorithm, which uses a replay buffer to store experiences
    and a target network to stabilize training. The agent interacts with the environment,
    collects experiences, and updates the Q-network based on sampled batches from the replay buffer.
    The training process includes epsilon-greedy action selection, experience replay, and periodic
    evaluation of the policy.

    Args:
        env: The environment to train the agent on.
        q_net: The Q-network to be trained.
        target_net: The target Q-network for stability.
        optimizer: Optimizer for the Q-network.
        episodes: Total number of training episodes.
        gamma: Discount factor for future rewards.
        batch_size: Size of the batch sampled from the replay buffer.
        buffer_capacity: Maximum capacity of the replay buffer.
        min_buffer_size: Minimum number of experiences in the buffer before training starts.
        epsilon_start: Initial value of epsilon for epsilon-greedy action selection.
        epsilon_end: Final value of epsilon after decay.
        epsilon_decay: Decay factor for epsilon.
        target_update: Frequency of updating the target network.
        run: WandB run object for logging.
        checkpoint_dir: Directory to save model checkpoints.
        device: Device to run the training on (CPU or GPU).

    Returns:
        episode_rewards: List of total rewards for each episode.
    """

    buffer = ReplayBuffer(capacity=buffer_capacity)
    epsilon = epsilon_start
    best_reward = float('-inf')
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(obs.unsqueeze(0))
                    action = q_values.argmax().item()

            next_obs_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Converti in tensore e sposta sul device
            next_obs = torch.tensor(next_obs_raw, dtype=torch.float32, device=device)

            # Salva su CPU nel buffer
            buffer.push(
                obs.cpu().numpy(),
                action,
                reward,
                next_obs.cpu().numpy(),
                done
            )

            # Avanza allo stato successivo
            obs = next_obs
            total_reward += reward

            if len(buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)


                q_values = q_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                    targets = rewards + gamma * max_next_q * (1 - dones)

                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if episode % eval_every == 0:
            avg_reward, avg_len = evaluate_policy(env, q_net, device=device)
            if run:
                run.log({
                    "eval_avg_reward": avg_reward,
                    "eval_avg_length": avg_len,
                    "epsilon": epsilon,
                    "loss": loss.item() if 'loss' in locals() else None
                })
            print(f"[Ep {episode}] Eval Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

            if checkpoint_dir:
                save_checkpoint("LATEST", q_net, target_net, optimizer, episode, checkpoint_dir)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_checkpoint("BEST", q_net, target_net, optimizer, episode, checkpoint_dir)

    return episode_rewards
