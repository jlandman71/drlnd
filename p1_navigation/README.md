[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Solution 

I have implemented Double Deep Q-Learning to get faster convergence than Vanilla Deep Q-Learning.

For implementing Double Deep Q-Learning I used Von Dollen's article "Investigating Reinforcement Learning Agents for
Continuous State Space Environments" [reference](https://arxiv.org/ftp/arxiv/papers/1708/1708.02378.pdf) as a starting point.

I used Von Dollen's recommended neural network architecture with first hidden layer of 128 nodes and second hidden layer of 256 nodes. Also I used the tanh activation function for connecting the first to the second hidden layer.

To implement Double learning the network for determining the next state's max actions (qnetwork_local) is seperated from the network that determines the next state's action values (qnetwork_target) (code snippet from dqn_agent.py):

   next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
   Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)

I have tested the implementation first on OpenAI Gym’s LunarLander-v2 to see if I could replicate Von Dollen's results.
After the results on OpenAI Gym’s LunarLander-v2 were satisfactory then I applied the same solution to the Banana Navigation project.
Then I had to decrease the optimizer's learning rate from 2e-3 to 2e-4 to get stable convergence.   

The end result is a fast convergence to the target score of 15.0 in 326 episodes.
