'''
Author: David Megli
Date: 2025-05-02
File: common.py
Description: Support functions
'''
import torch

def evaluate_policy(env, q_net, episodes=5):
    q_net.eval()
    total_rewards = []
    total_lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        reward_sum = 0
        steps = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = q_net(obs_tensor).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_sum += reward
            steps += 1

        total_rewards.append(reward_sum)
        total_lengths.append(steps)

    q_net.train()
    return sum(total_rewards) / len(total_rewards), sum(total_lengths) / len(total_lengths)
