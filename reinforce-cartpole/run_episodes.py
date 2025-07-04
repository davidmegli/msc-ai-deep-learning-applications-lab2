import argparse
import gymnasium
from gymnasium.wrappers import RecordVideo
from common import run_episode
import os
from networks import *

def main(args):
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # If not specified, use the first file in "model"
        directory = "model"
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            raise FileNotFoundError("No checkpoint found in 'model' directory.")
        checkpoint_path = os.path.join(directory, files[0])
        print(f"✅ Using default checkpoint: {checkpoint_path}")

    # Choosing the environment
    if args.env.lower() == 'cartpole':
        environment = 'CartPole-v1'
    elif args.env.lower() == 'lunarlander':
        environment = 'LunarLander-v3'
    else:
        raise ValueError(f'Unknown environment {args.env}')

    # Loading policy network
    env = gymnasium.make(environment)
    policy = PolicyNet(env, n_hidden=args.depth, width=args.width)
    policy = load_checkpoint(checkpoint_path, policy)

    # Creating environment for rendering or recording
    if args.record:
        os.makedirs("videos", exist_ok=True)
        env_render = gymnasium.make(environment, render_mode='rgb_array')
        env_render = RecordVideo(
            env_render,
            video_folder='videos',
            episode_trigger=lambda ep: True,
            name_prefix=environment.lower(),
            video_length=10000  # Salva episodi lunghi
        )
        print("📽️ Recording mode ON: saving episodes to /videos")
    else:
        env_render = gymnasium.make(environment, render_mode='human')
        print("👁️ Visual mode ON: rendering live episodes")

    # Esegui le simulazioni
    for _ in range(args.n):
        run_episode(env_render, policy)

    env_render.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Run Cartpole or LunarLander episodes using a pretrained PolicyNet.')
    parser.add_argument('--n', type=int, default=10, help='Number of episodes')
    parser.add_argument('--env', type=str, default='cartpole', help='Environment: cartpole or lunarlander')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--record', action='store_true', help='If set, saves episodes to video instead of rendering live')
    parser.add_argument('--width', type=int, default=256, help='Width of the loaded network')
    parser.add_argument('--depth', type=int, default=2, help='Depth of the loaded network')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
