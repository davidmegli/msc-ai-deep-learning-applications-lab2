'''
Author: David Megli
Date: 2025-05-02
File: common.py
Description: Support functions
'''
import torch

def evaluate_policy(env, q_net, episodes=5, device='cpu'):
    q_net.eval()
    total_rewards = []
    total_lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        reward_sum = 0
        steps = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
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

import torch
import os

def save_checkpoint(name, q_net, target_net, optimizer, step, path):
    """
    Save the current Q-network, target network, and optimizer state.
    
    Args:
        name (str): Name of the checkpoint, e.g., 'LATEST', 'BEST'.
        q_net (torch.nn.Module): The main Q-network.
        target_net (torch.nn.Module): The target Q-network.
        optimizer (torch.optim.Optimizer): Optimizer for Q-network.
        step (int): Current training step or episode.
        path (str): Path where the checkpoint will be saved.
    """
    torch.save({
        'step': step,
        'q_net_state_dict': q_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(path, f'checkpoint-{name}.pt'))


def load_checkpoint(name, q_net, target_net, optimizer, path, device='cpu'):
    """
    Load a checkpoint into Q-network, target network, and optimizer.

    Args:
        name (str): Name of the checkpoint, e.g., 'LATEST', 'BEST'.
        q_net (torch.nn.Module): The main Q-network to load into.
        target_net (torch.nn.Module): The target Q-network to load into.
        optimizer (torch.optim.Optimizer): The optimizer to load into.
        path (str): Path where the checkpoint is located.
        device (str): Device to map the model ('cpu' or 'cuda').

    Returns:
        int: The step or episode number saved in the checkpoint.
    """
    checkpoint = torch.load(os.path.join(path, f'checkpoint-{name}.pt'), map_location=device)
    q_net.load_state_dict(checkpoint['q_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def load_policy(path, q_net):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    q_net.load_state_dict(checkpoint['q_net_state_dict'])
    print(f"✅ Loaded policy from {path}")
    return q_net
