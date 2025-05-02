'''
Author: David Megli
Date: 2025-05-05
File: dqn.py
Description: Main Deep Q-Network Algorithm (training loop)
'''
import torch
import torch.nn.functional as F
import random
from collections import namedtuple
from common import run_episode, evaluate_policy
from replay_buffer import ReplayBuffer

def dqn(env, q_net, target_net, optimizer, episodes=500, gamma=0.99, batch_size=64,
        buffer_capacity=100_000, min_buffer_size=1000, epsilon_start=1.0,
        epsilon_end=0.05, epsilon_decay=0.995, target_update=100, eval_every=50, run=None):

    buffer = ReplayBuffer(capacity=buffer_capacity)
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(obs.unsqueeze(0))
                    action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(obs.numpy(), action, reward, next_obs, done)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            total_reward += reward

            if len(buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

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
            avg_reward, avg_len = evaluate_policy(env, q_net)
            if run:
                run.log({"eval_avg_reward": avg_reward, "eval_avg_length": avg_len})
            print(f"[Ep {episode}] Eval Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    return episode_rewards
