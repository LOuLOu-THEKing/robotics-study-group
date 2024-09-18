# Vanilla Policy Gradient (VPG) Implementation

## TLDR
A simple PyTorch Reinforcement Learning model where an agent learns to move towards food while avoiding enemies on a 2D grid, based on log-probabilities and policy updates using backpropagation through time.

## Short Overview of VPG

- Stochastic policy: $\pi(a|s,\theta)$
  where $\pi$ represents the probability of taking action $a$ given state $s$ and parameters $\theta$.

- The objective is to change parameters $\theta$ to maximize expected rewards.

- Unlike TD learning, there's no need for calculating Q-value or V-value.

- Better suited for continuous state spaces.


## VPG

The goal of PG methods is to maximize the expected return:

$J(\theta) = \mathbb{E}{\tau \sim \pi\theta}[\sum_{t=0}^T R(s_t, a_t)]$


where $\tau$ represents a trajectory, and $R(s_t, a_t)$ is the reward at time $t$.

The policy gradient theorem gives us:
$\nabla_\theta J(\theta) = \mathbb{E}{\tau \sim \pi\theta}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R_t]$

where $R_t = \sum_{t'=t}^T R(s_{t'}, a_{t'})$ is the return from time $t$.


It is fundamental algorithm in RL that provides a way to compute the gradient of the expected return with respect to the policy parameters. This is really nice because it allows us to optmize the policy directly, without needing to estimate action-values.


### The formal statement

The intuition behind this is that we need to increase the probability of actions that lead to good outcomes and decrease the probability of actions that lead to bad outcomes.


Let $\pi_\theta(a|s)$ be a stochastic policy parametrized by $\theta$, and let $J(\theta)$ be the expected return when following this policy. The policy gradient theorem states that:

$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R_t]$

where:
- $\tau$ is a trajectory $(s_0, a_0, r_0, s_1, a_1, r_1, ...)$
- $R_t = \sum_{t'=t}^T r_{t'}$ is the return from time $t$


1. **Log-probability gradient**: $\nabla_\theta \log \pi_\theta(a_t|s_t)$
   - This term indicates how the log probability of taking action $a_t$ in state $s_t$ changes with respect to the policy parameters.

2. **Return**: $R_t$
   - This is the cumulative reward from time $t$ onwards.

3. **Expectation**: $\mathbb{E}_{\tau \sim \pi_\theta}$
   - This indicates that we're averaging over all possible trajectories under the current policy.

### Interpretation

The theorem tells us that to increase the expected return, we should move the policy parameters in a direction that:
1. Increases the probability of actions that led to high returns.
2. Decreases the probability of actions that led to low returns.

The magnitude of the update is proportional to the return, so actions that led to much better (or worse) outcomes have a larger impact on the policy update.

The policy gradient theorem works because:
1. It directly optimizes the quantity we care about (expected return).
2. It doesn't require estimating action-values, which can be difficult in large state spaces.
3. It naturally handles continuous action spaces.

While awesome, the policy gradient theorem has some limitations:
1. High variance: The returns can vary a lot, leading to noisy gradients.
2. Sample inefficiency: It often requires many samples to estimate the gradient accurately.

These limitations have led to various improvements and extensions.

## Environment

The environment in this problem is a simple 2D grid-based world where there are 3 blobs (player, food, and enemy). The objective of the RL agent is to move towards the food blob while avoiding the enemy blob by choosing appropriate actions based on its current state.

The environment is represented as a numpy array of size SIZE x SIZE where each cell can contain one of 3 blobs. The input for the policy network consists of the relative positions of these 3 blobs in the current state.

## Reinforcement Learning Approach

In this code, we use PyTorch to build a simple policy gradient RL model with a neural network that represents our policy function.

This network takes our state representations as input and outputs a probability distribution over possible actions. Mathematically, for a state $s$:

$\pi_\theta(a|s) = \text{softmax}(f_\theta(s))$

where $f_\theta$ is our neural network.

The agent then selects an action based on these probabilities using the Categorical distribution's sample method.


## Cost Function in Policy Gradient Methods

In policy gradient methods, our objective is to maximize the expected return. However, in practice/implementation, we often minimize a loss/cost function that, when minimized, maximizes our expected return. The cost function for policy gradient methods is defined as:

$J(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T R(s_t, a_t)]$

Here
- $\theta$ represents the parameters of our policy, $\tau$ is a trajectory and $R(s_t, a_t)$ is the reward at time $t$.

The negative sign is crucial because we typically **perform gradient descent to minimize this function**, which effectively maximizes our expected return.

## Detailed Algorithm Steps

1. **Initialize the Policy Network**: 
   We start by initializing our policy network $\pi_\theta(a|s)$ with random weights $\theta$.

2. **Collect Trajectories**:
   For each episode:
   a. Initialize the environment and get the initial state $s_0$.
   b. For each time step $t$ in the episode:
      - Sample action $a_t \sim \pi_\theta(a|s_t)$
      - Execute $a_t$ and observe next state $s_{t+1}$ and reward $r_t$
      - Store $(s_t, a_t, r_t)$ in the current trajectory
   c. End the episode when a terminal state is reached

3. **Compute Returns**:
   For each trajectory:
   a. Initialize $G_T = 0$ (return at the final step)
   b. For $t = T-1$ to 0:
      - Compute $G_t = r_t + \gamma G_{t+1}$
   This gives us the discounted return for each time step.

4. **Normalize Returns** (Optional):
   To reduce variance, we often normalize the returns:
   $G_t' = \frac{G_t - \mu(G)}{\sigma(G)}$
   where $\mu(G)$ and $\sigma(G)$ are the mean and standard deviation of all returns.

5. **Compute Policy Gradient**:
   The policy gradient is computed as:
   $\nabla_\theta J(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) G_t']$

   In practice, we approximate this expectation with our sampled trajectories:
   $\nabla_\theta J(\theta) \approx -\frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) G_t'^i$

   where $N$ is the number of trajectories.

6. **Update Policy Parameters**:
   We update our policy parameters using gradient descent:
   $\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)$

   where $\alpha$ is the learning rate.

7. **Repeat**:
   Go back to step 2 and repeat for many episodes until convergence or a maximum number of episodes is reached.

## Implementation

```python
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def update_policy(log_probs, rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G  # 0.99 is the discount factor
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalization
    
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)  # Negative because we're doing gradient ascent
    
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
```

![VPG-training](media/vpg_training.png)
<br>
![VPG-moving](media/vpg_moving.gif)


## Implementation Details

- Steps 1-3 are implemented in the main training loop and the `select_action` function.
- Step 4 (normalization) is done in the `update_policy` function.
- Steps 5-6 are implemented in the `update_policy` function, where we compute the policy loss and perform backpropagation.

The key part is in the `update_policy` function:

```python
policy_loss = []
for log_prob, R in zip(log_probs, returns):
    policy_loss.append(-log_prob * R)
```

This line implements the policy gradient. The negative sign is present because we're doing gradient descent to minimize the negative expected return (which is equivalent to maximizing the expected return).

The `backward()` call computes the gradients, and the `optimizer.step()` updates the parameters, implementing step 6 of our algorithm.

By iterating this process over many episodes, the policy gradually improves, learning to maximize the expected return in the given environment.

## Mathematical Explanation of the Update Step

1. We calculate the returns $G_t$ for each timestep:

   $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$

   where $\gamma$ is the discount factor (0.99 in this code).

2. We normalize the returns to reduce variance:

   $G_t' = \frac{G_t - \mu(G)}{\sigma(G)}$

3. The policy loss for each step is calculated as:

   $L_t = -\log \pi_\theta(a_t|s_t) G_t'$

4. The total loss is the sum of these individual losses:

   $L = \sum_t L_t$

5. We then use backpropagation to compute $\nabla_\theta L$ and update our policy parameters:

   $\theta \leftarrow \theta - \alpha \nabla_\theta L$

   where $\alpha$ is the learning rate.


This implementation follows the core principles of the REINFORCE algorithm, a fundamental policy gradient method in reinforcement learning.


full code here [vpg.py](./code/vpg.py)

