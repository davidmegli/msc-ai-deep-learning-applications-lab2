import argparse
import wandb
import gymnasium
from networks import PolicyNet
from reinforce import reinforce
from common import run_episode

def parse_args():
    """The argument parser for the main training script."""
    parser = argparse.ArgumentParser(description='A script implementing REINFORCE on the Cartpole environment.')
    parser.add_argument('--project', type=str, default='DLA2025-Cartpole', help='Wandb project to log to.')
    parser.add_argument('--baseline', type=str, default='none', help='Baseline to use (none, std, value)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate agent every N episodes')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes to use in each evaluation')
    parser.add_argument('--visualize', action='store_true', help='Visualize final agent')
    parser.add_argument('--env', type=str, default='cartpole', help='environment to use (cartpole, lunarlander)')
    parser.set_defaults(visualize=True)
    return parser.parse_args()


# Main entry point.
if __name__ == "__main__":
    # Get command line arguments.
    args = parse_args()

    # Initialize wandb with our configuration parameters.
    run = wandb.init(
        project=args.project,
        config={
            'learning_rate': args.lr,
            'baseline': args.baseline,
            'gamma': args.gamma,
            'num_episodes': args.episodes
        }
    )

    maxlen=None
    environment = ''
    if args.env.lower() == 'cartpole':
        environment = 'CartPole-v1'
        maxlen=500
    elif args.env.lower() == 'lunarlander':
        environment = 'LunarLander-v3'
        maxlen=500
    else:
        raise ValueError(f'Unknown environment {args.env}')

    # Instantiate the Cartpole environment (no visualization).
    env = gymnasium.make(environment)

    hidden_layers = 2
    width = 256
    # Make a policy network.
    policy = PolicyNet(env, n_hidden=hidden_layers, width=width)
    # If using value baseline, also create a value network
    if args.baseline == 'value':
        from networks import ValueNet
        value_net = ValueNet(env, n_hidden=hidden_layers, width=width)
    else:
        value_net = None


    # Train the agent.
    reinforce(
        policy,
        env,
        run,
        lr=args.lr,
        baseline=args.baseline,
        num_episodes=args.episodes,
        gamma=args.gamma,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        value_net=value_net,
        maxlen=maxlen)

    # And optionally run the final agent for a few episodes.
    if args.visualize:
        env_render = gymnasium.make(environment, render_mode='human')
        for _ in range(10):
            run_episode(env_render, policy,maxlen)

        # Close the visualization environment.
        env_render.close()

    # Close the Cartpole environment and finish the wandb run.
    env.close()
    run.finish()
