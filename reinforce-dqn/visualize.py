'''
Author: David Megli
Date: 2025-05-02
File: visualize.py
Description: Script to visualize simulations from a pretrained checkpoint or save them as videos
'''
import os
import argparse
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from networks import QNetwork
from common import load_policy

def visualize(env_id, checkpoint_path, hidden_dim=128, episodes=5, record_video=False):
    # Se vogliamo salvare video, impostiamo il wrapper
    if record_video:
        os.makedirs("videos", exist_ok=True)
        env = gym.make(env_id, render_mode='rgb_array')
        env = RecordVideo(env, video_folder='videos', episode_trigger=lambda ep: True, name_prefix=env_id)
    else:
        env = gym.make(env_id, render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim, hidden_dim)
    q_net = load_policy(checkpoint_path, q_net)
    q_net.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            with torch.no_grad():
                q_values = q_net(obs.unsqueeze(0))
                action = q_values.argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward
            steps += 1

        print(f"[Episode {ep + 1}] Reward: {total_reward:.2f}, Steps: {steps}")

    env.close()
    if record_video:
        print("📽️ Videos saved to: videos/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['cartpole', 'lunarlander'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--record_video', action='store_true', help="Set this flag to record videos instead of rendering live.")
    args = parser.parse_args()

    env_map = {
        'cartpole': 'CartPole-v1',
        'lunarlander': 'LunarLander-v3'
    }

    visualize(env_map[args.env], args.checkpoint, args.hidden_dim, args.episodes, args.record_video)
