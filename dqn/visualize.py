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

from networks import QNetwork, CNNQNetwork
from common import load_policy
from preprocess import FrameStacker

def visualize(env_id, checkpoint_path, hidden_dim=128, episodes=5, record_video=False):
    is_carracing = 'CarRacing' in env_id

    if record_video:
        os.makedirs("videos", exist_ok=True)
        render_mode = 'rgb_array'
    else:
        render_mode = 'human'

    if is_carracing:
        base_env = gym.make(env_id, render_mode=render_mode, continuous=False).env
        env = FrameStacker(base_env, k=3)  # Uso 3 frame come nel training
    else:
        env = gym.make(env_id, render_mode=render_mode)

    if is_carracing:
        input_channels = env.observation_space.shape[0]  # e.g., 3
        q_net = CNNQNetwork(input_channels=input_channels, action_dim=env.action_space.n, hidden_dim=hidden_dim)
    else:
        state_dim = env.observation_space.shape[0]
        q_net = QNetwork(state_dim, env.action_space.n, hidden_dim)


    q_net = load_policy(checkpoint_path, q_net)
    q_net.eval()

    if record_video:
        env = RecordVideo(env, video_folder='videos', episode_trigger=lambda ep: True, name_prefix=env_id, video_length=0)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = q_net(obs_tensor).argmax().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"[Episode {ep + 1}] Reward: {total_reward:.2f}, Steps: {steps}")

    env.close()
    if record_video:
        print("📽️ Videos saved to: videos/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['cartpole', 'lunarlander', 'carracing'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--record_video', action='store_true', help="Set this flag to record videos instead of rendering live.")
    args = parser.parse_args()

    env_map = {
        'cartpole': 'CartPole-v1',
        'lunarlander': 'LunarLander-v3',
        'carracing': 'CarRacing-v3'
    }

    visualize(env_map[args.env], args.checkpoint, args.hidden_dim, args.episodes, args.record_video)
