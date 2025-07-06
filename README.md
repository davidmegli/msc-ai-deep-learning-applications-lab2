# David Megli - Deep Learning Applications Lab 2

This project is an implementation of Lab 2 for the Deep Learning Applications course in the Master's Degree in Artificial Intelligence at the University of Florence.

Here are the requirements, the exercises descriptions and results, and the instructions to train and test the trained models.

## Requirements

- Python 3.6+
- PyTorch
- Gymnasium
- WandB (Weights and Biases)

# Exercises

## Exercise 1
My version of the reinforce algorithm can be found in ```reinforce-cartpole/reinforce.py```. I used as a baseline the professor's Reinforce function.
## Exercise 2
### Usage
### Description
The standardized baseline was already implemented in ```reinforce-cartpole/reinforce.py```. I
I implemented the ValueNet in ```reinforce-cartpole/networks.py```, which is a deep neural network with the same structure as the PolicyNet, but is used to estimate the value function, which can be used as a baseline in the Reinforce algorithm.
Th

### Results

![Reinforce Cartpole baseline comparison](assets/reinforce-cartpole-baseline-comparison.png)
## Exercise 3
### Exercise 3.1
![Reinforce Lunar Lander avg episode length](assets/reinforce-lunar-lander-length.png)
![Reinforce Lunar Lander avg reward](assets/reinforce-lunar-lander-reward.png)

### Exercise 3.2 & 3.3
![DQN episode length](assets/DQN-length.png)
![DQN reward](assets/DQN-reward.png)

## Reinforce
### Training
Run the main training script with the following command:
```bash
python main.py [OPTIONS]
```

- `--project`: Name of the WandB project (default: `DLA2025-Cartpole`)
- `--baseline`: Type of baseline to use (options: `none`, `std`, `value`, default: `none`)
- `--gamma`: Discount factor for future rewards (default: `0.99`)
- `--lr`: Learning rate for the optimizer (default: `1e-3`)
- `--episodes`: Number of training episodes (default: `1000`)
- `--eval_every`: Evaluate agent every N episodes (default: `50`)
- `--eval_episodes`: Number of episodes to use in each evaluation (default: `10`)
- `--visualize`: Set flag to visualize the final agent (default: `False`)
- `--env`: Environment to use (default: `cartpole`, options: `cartpole`, `lunarlander`)
#### Example
To train the agent without using a baseline and visualize the final agent, you can run:

```bash
wandb login
python main.py --baseline none --visualize
```


## Results
You can find the qualitative results in this [YouTube video](https://youtu.be/pdkXgKGKqQI)