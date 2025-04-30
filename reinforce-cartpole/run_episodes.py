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
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    # first file name
    if files:
        first_file = files[0]
        print("First file:", first_file)
    else:
        print("No files found in the directory.")
    first_file = os.path.join(directory, first_file)
    # loading checkpoint

    # Instantiate the Cartpole environment (no visualization).
    env = gymnasium.make('CartPole-v1')

    # Make a policy network.
    policy = PolicyNet(env)

    policy = load_checkpoint(first_file, policy)
    env_render = gymnasium.make('CartPole-v1', render_mode='human')
    for _ in range(args.n):
        run_episode(env_render, policy)

    # Close the visualization environment.
    env_render.close()

def parse_args():
    parser = argparse.ArgumentParser(description='A script running Cartpole episodes using a pretrained model')
    parser.add_argument('--n', type=int, default=10, help='Number of episodes')
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    main(args)