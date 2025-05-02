import argparse
import torch
import gymnasium as gym

from networks import QNetwork
from common import load_policy

def visualize(env_id, checkpoint_path, hidden_dim=128, episodes=5):
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

        while not done:
            with torch.no_grad():
                q_values = q_net(obs.unsqueeze(0))
                action = q_values.argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward

        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['cartpole', 'lunarlander'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    env_map = {
        'cartpole': 'CartPole-v1',
        'lunarlander': 'LunarLander-v3'
    }

    visualize(env_map[args.env], args.checkpoint, args.hidden_dim, args.episodes)
