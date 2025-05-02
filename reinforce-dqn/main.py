'''
Author: David Megli
Date: 2025-05-02
File: main.py
Description: main script to run DQN
'''
import argparse
import gymnasium as gym
import torch
import wandb

from dqn import train_dqn
from networks import QNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole', choices=['cartpole', 'lunarlander'])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.995)
    parser.add_argument('--target_update_freq', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--project', type=str, default='DQN-Project')
    return parser.parse_args()


def main():
    args = parse_args()

    env_id = 'CartPole-v1' if args.env == 'cartpole' else 'LunarLander-v3'
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim, args.hidden_dim)
    target_net = QNetwork(state_dim, action_dim, args.hidden_dim)
    target_net.load_state_dict(q_net.state_dict())  # Sync initially
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    run = wandb.init(
        project=args.project,
        config=vars(args),
        name=f"DQN-{args.env}-hd{args.hidden_dim}-lr{args.lr}"
    )

    train_dqn(
        env=env,
        q_net=q_net,
        target_net=target_net,
        optimizer=optimizer,
        episodes=args.episodes,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay=args.eps_decay,
        target_update=args.target_update_freq,
        eval_every=args.eval_every,
        run=run,
        checkpoint_dir=run.dir
    )

    run.finish()


if __name__ == '__main__':
    main()
