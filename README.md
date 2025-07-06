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
#### Description
The standardized baseline was already implemented in ```reinforce-cartpole/reinforce.py```. 
I implemented the ValueNet in ```reinforce-cartpole/networks.py```, which is a deep neural network with the same structure as the PolicyNet, but is used to estimate the value function, which can be used as a baseline in the Reinforce algorithm.
#### Results
From the next plot we can appreciate the quantitative comparison between the different baselines evaluated on the Cartpole environment. The reported metric is the average episode length evaluated every N training episodes on M episodes.
We can see that without any baseline the network is not even able to learn to keep the pole erect after 5000 episodes, insted we see that after about 2000 episodes the average episode length even gets worse!
The std baseline, which consists of subtracting the average return and dividing by the variance of the returns, significantly improves the performance, enabling the network to learn to keep the pole erect.
The Value baseline, which uses the Value Network to estimate the value function to use it as a baseline, improves even more the performance, making the network converge faster to the maximum episode length and increasing stability during training.
![Reinforce Cartpole baseline comparison](assets/reinforce-cartpole-baseline-comparison.png)
## Exercise 3
### Exercise 3.1
In this exercises I solved the Lunar Lander Environment using the Reinforce algorithm, the PolicyNet and the ValueNet implemented in the previous exercise.

In my first attempts I tried to train the network for 5000 or 10000 episodes, but the network didn't seem to learn how to make the spaceship land. Indeed it seemed like the network just learned to keep the spaseship flying, to not receive the big negative reward to crash on the ground, and the learning curve had a plateau, as you can see from the following 2 plots, reporting the average evaluated length and reward.
The plateau in the average length, which increases instead of decreasing, shows indeed that the network learns to keep the spaceship flying as long as possible, using as less as possible fuel. Also, the episode is considered solved if the score is higher than 200, and the reward for landing on the objective is 100, but in the second plot we can see that the reward doesn't even reach 50, while with lower learning rate it remains negative.
![Reinforce Lunar Lander avg episode length bad](assets/reinforce-lunar-lander-length-bad.png)
![Reinforce Lunar Lander avg reward bad](assets/reinforce-lunar-lander-reward-bad.png)
These are just a few examples of the attempts I made, with different settings. As you can see I also tried to lower the learning rate, hoping that instead of just learning to keep the spaseship flying, the network would have had more time to explore different actions and discovered the big reward it would get for landing on the objective without crashing.

After these failed attempts, I thought about letting the network run for many more episodes, hoping that it would sooner or later land on the objective (I also tried to lower the learning rate).
And that's actually what happened!
As you can see from the following 2 plots, after about 32000 episodes there is a drop of the episode length and an upward spike in the reward.
This probably means that around the episode 32000, in this particular case, the spaceship landed for the first time, and the network started to learn how to get an higher reward.
Indeed from this point the average episode length drops from 500 to around 200, and the reward converges to 270, surpassing 200, considered as solved, from around episode 39950, so after about 7000 episodes from the (supposed) first landing.
![Reinforce Lunar Lander avg episode length](assets/reinforce-lunar-lander-length.png)
![Reinforce Lunar Lander avg reward](assets/reinforce-lunar-lander-reward.png)
Note: I probably could have used a higher learning rate, and just let the network train for enough episodes.

### Exercise 3.2
In this exercise I implemented the Deep Q-Learning algorithm to solve Cartpole and Lunar Lander.
![DQN episode length](assets/DQN-length.png)
![DQN reward](assets/DQN-reward.png)

## Qualitative Results
Some qualitative results can be found in the ```videos``` folder inside the ```reinforce-cartpole``` and ```dqn``` folders.
However these results have been reported in [this YouTube video](https://youtu.be/pdkXgKGKqQI)

## Reinforce
### Training
Run the main training script in the ```reinforce-cartpole``` folder with the following command:
```bash
python main.py [OPTIONS]
```

#### Options
- `--project`: Name of the WandB project (default: `DLA2025-Cartpole`)
- `--baseline`: Type of baseline to use (options: `none`, `std`, `value`, default: `none`)
- `--gamma`: Discount factor for future rewards (default: `0.99`)
- `--lr`: Learning rate for the optimizer (default: `1e-3`)
- `--episodes`: Number of training episodes (default: `1000`)
- `--eval_every`: Evaluate agent every N episodes (default: `50`)
- `--eval_episodes`: Number of episodes to use in each evaluation (default: `10`)
- `--visualize`: Set flag to visualize the final agent (default: `False`)
- `--env`: Environment to use (options: `cartpole`, `lunarlander`, default: `cartpole`)
#### Example
To train the agent without using a baseline and visualize the final agent, you can run:

```bash
wandb login
python main.py --baseline none --visualize
```
### Test
Run the test in the ```reinforce-cartpole``` folder with the following command:
```bash
python run_episodes.py [OPTIONS]
```
#### Options
- `--n`: Number of episodes (default: `10`)
- `--env`: Environment to use (options: `cartpole`, `lunarlander`, default: `cartpole`)
- `--checkpoint`: Path to the checkpoint
- `--record`: If set, saves episodes to video instead of rendering live
- `--width`: Width of the loaded network (default: `256`)
- `--depth`: Evaluate agent every N episodDepth of the loaded network (default: `2`)
Note: the sizes of the trained networks were 2x256 and 1x128.
#### Examples
```bash
python run_episodes.py --5 --env cartpole --checkpoint model/cartpole-checkpoint-BEST.pt 
```
```bash
python run_episodes.py --5 --env lunarlander --checkpoint model/lunarlander-checkpoint-BEST.pt 
```
## Deep Q-Network

### Training
Run the main training script in the ```dqn``` folder with the following command:
```bash
python main.py [OPTIONS]
```

##### Options
- `--env`: Environment to use (options: `cartpole`, `lunarlander`, `carracing`, default: `cartpole`)
- `--project`: Name of the WandB project (default: `DQN-Project`)
- `--episodes`: Number of training episodes (default: `1000`)
- `--lr`: Learning rate for the optimizer (default: `1e-3`)
- `--gamma`: Discount factor for future rewards (default: `0.99`)
- `--batch_size`: Size of the batch sampled from the replay buffer (default: `50`)
- `--buffer_size`: Maximum capacity of the replay buffer (default: `64`)
- `--hidden_dim`: Hidden dimension of the Q-network (default: `100000`)
- `--eval_every`: Evaluate agent every N episodes (default: `128`)
- `--eps_start`: Initial value of epsilon for epsilon-greedy action selection
- `--eps_end`: Final value of epsilon after decay
- `--eps_decay`: Decay factor for epsilon
- `--target_update_freq`: Frequency of updating the target network
- `--eval_every`: Frequency of evaluation during training
##### Example
To train the agent without using a baseline and visualize the final agent, you can run:

```bash
wandb login
python main.py --env carracing
```

### Test
Run the test in the ```dqn``` folder with the following command:
```bash
python visualize.py [OPTIONS]
```
##### Options
- `--env`: Environment to use (options: `cartpole`, `lunarlander`, `carracing`)
- `--checkpoint`: Path to the checkpoint
- `--hidden_dim`: Size of the hidden layers (Default = `128`)
- `--episodes`: Number of episodes to visualize (Default = `5`)
- `--record_video`: If set, saves episodes to video instead of rendering live

##### Examples
```bash
python visualize.py --env cartpole --checkpoint model/cartpole-checkpoint-BEST.pt  --episodes 5
```
```bash
python visualize.py --env lunarlander --checkpoint model/lunarlander-checkpoint-BEST.pt  --episodes 5
```
```bash
python visualize.py --env carracing --checkpoint model/carracing-checkpoint-BEST.pt  --episodes 5
```