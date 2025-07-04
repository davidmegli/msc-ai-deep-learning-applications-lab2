# David Megli - Deep Learning Applications Lab 2

This project is an implementation of Lab 2 for the Deep Learning Applications course in the Master's Degree in Artificial Intelligence at the University of Florence.

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


## Results
You can find the qualitative results in this [YouTube video](https://youtu.be/pdkXgKGKqQI)