import argparse
import gymnasium
from common import run_episode
import os
from networks import *

def main(args):
    # model folder name
    directory = "model"
    files = os.listdir(directory)
    # Filter out only files 
    '''files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    # first file name
    if files:
        first_file = files[0]
        print("First file:", first_file)
    else:
        print("No files found in the directory.")
    first_file = os.path.join(directory, first_file)'''
    first_file = args.checkpoint
    # loading checkpoint

    environment = ''
    if args.env.lower() == 'cartpole':
        environment = 'CartPole-v1'
    elif args.env.lower() == 'lunarlander':
        environment = 'LunarLander-v3'
    else:
        raise ValueError(f'Unknown environment {args.env}')
    env = gymnasium.make(environment)

    # Make a policy network.
    policy = PolicyNet(env)

    policy = load_checkpoint(first_file, policy)
    env_render = gymnasium.make(environment, render_mode='human')
    for _ in range(args.n):
        run_episode(env_render, policy)

    # Close the visualization environment.
    env_render.close()

def parse_args():
    parser = argparse.ArgumentParser(description='A script running Cartpole episodes using a pretrained model')
    parser.add_argument('--n', type=int, default=10, help='Number of episodes')
    parser.add_argument('--env', type=str, default='cartpole', help='environment to use (cartpole, lunarlander)')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    main(args)