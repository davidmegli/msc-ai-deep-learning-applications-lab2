import torch
import wandb
from networks import save_checkpoint
from common import run_episode, compute_returns, select_action

# è gia implementato Standardization Baseline (- media diviso varianza)
# 1. No Baseline
# 2. Standardization Baseline
# 3. Value Baseline (esercizio 3)
def reinforce(policy, env, run, gamma=0.99, lr=0.02, baseline='std',
              num_episodes=10, eval_every=10, eval_episodes=5):
    """
    REINFORCE with optional evaluation every N episodes.

    Args:
        policy: The policy network to be trained.
        env: The environment in which the agent operates.
        run: wandb run object.
        gamma: Discount factor for future rewards.
        lr: Learning rate.
        baseline: Type of baseline ('none', 'std').
        num_episodes: Total number of training episodes.
        eval_every: Evaluate agent every N episodes.
        eval_episodes: Number of evaluation episodes per evaluation cycle.

    Returns:
        running_rewards: List of running rewards.
        eval_metrics: List of dicts with evaluation stats (reward, length).
    """
    # Check for valid baseline (should probably be done elsewhere).
    if baseline not in ['none', 'std']:
        raise ValueError(f'Unknown baseline {baseline}')

    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # Track episode rewards in a list.
    running_rewards = [0.0]
    best_return = 0.0
    # Track evaluation metrics
    eval_metrics = []

    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        # New dict for the wandb log for current iteration.
        log = {}

        # Run an episode of the environment, collect everything needed for policy update.
        observations, actions, log_probs, rewards = run_episode(env, policy)

        # Compute the discounted reward for every step of the episode. 
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)

        # Keep a running average of total discounted rewards for the whole episode.
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        # Log some stuff.
        log['episode_length'] = len(returns)
        log['return'] = returns[0]

        # Checkpoint best model.
        if running_rewards[-1] > best_return:
            save_checkpoint('BEST', policy, opt, wandb.run.dir)

        # Basline returns.
        if baseline == 'none':
            base_returns = returns
        elif baseline == 'std': # Standardization: subtracting mean, dividing by std
            base_returns = (returns - returns.mean()) / returns.std()

        # Make an optimization step on the policy network.
        opt.zero_grad()
        policy_loss = (-log_probs * base_returns).mean()
        policy_loss.backward()
        opt.step()

        # Log the current loss
        log['policy_loss'] = policy_loss.item()

        # Evaluate agent every N episodes
        if episode % eval_every == 0:
            policy.eval()
            total_rewards = []
            episode_lengths = []

            for _ in range(eval_episodes):
                obs, info = env.reset()
                rewards = []
                done = False
                steps = 0
                while not done and steps < 500:
                    obs_tensor = torch.tensor(obs)
                    action, _ = select_action(env, obs_tensor, policy)
                    obs, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    rewards.append(reward)
                    steps += 1

                total_rewards.append(sum(rewards))
                episode_lengths.append(steps)

            avg_reward = sum(total_rewards) / eval_episodes
            avg_length = sum(episode_lengths) / eval_episodes
            log['eval_avg_reward'] = avg_reward
            log['eval_avg_length'] = avg_length
            eval_metrics.append({'episode': episode, 'reward': avg_reward, 'length': avg_length})

            policy.train()  # Switching back to training mode

        # Logging to wandb
        run.log(log)

        # Print running reward and (optionally) render an episode after every 100 policy updates.
        if not episode % 100:
            print(f'Running reward @ episode {episode}: {running_rewards[-1]}')

    # Return the running rewards.
    policy.eval()
    return running_rewards
