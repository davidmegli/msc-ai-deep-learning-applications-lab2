# A starter project for REINFORCE applied to Cartpole

The repository contains a very simple, inefficient, and likely *buggy* implementation of REINFORCE. I tried to keep it simple, but at the same time implement logging via Weights and Biases and a minimal command line interface. REINFORCE is a policy gradient method that optimizes the policy directly. This implementation is built on top of the Farama Gymnasium library for creating the CartPole environment and weighs the benefits of using a standardization baseline to help reduce variance in training.

## Requirements

- Python 3.6+
- PyTorch
- Gymnasium
- WandB (Weights and Biases)

## Installation

Clone the repository and install the requirements:

```bash
git clone https://gitlab.com/bagdanov/reinforce-cartpole.git
cd reinforce-cartpole
conda create --name reinforce-cartpole --file requirements.txt
conda activate reinforce-cartpole
```

If you prefer, you can directly create the environment with `conda` like so:

```bash
conda create -n reinforce-cartpole -c conda-forge gymnasium pytorch matplotlib pygame wandb
conda activate reinforce-cartpole
```

## Usage

Run the main training script with the following command:

```bash
python main.py [OPTIONS]
```

### Options

- `--project`: Name of the WandB project (default: `DLA2025-Cartpole`)
- `--baseline`: Type of baseline to use (options: `none`, `std`, default: `none`)
- `--gamma`: Discount factor for future rewards (default: `0.99`)
- `--lr`: Learning rate for the optimizer (default: `1e-3`)
- `--episodes`: Number of training episodes (default: `1000`)
- `--visualize`: Set flag to visualize the final agent (default: `False`)

### Example

To train the agent without using a baseline and visualize the final agent, you can run:

```bash
wandb login
python main.py --baseline none --visualize
```
