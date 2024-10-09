## Overview

- What is RL?

- Why is RL Better?

- Elements of RL

- K-Armed bandits and maths
### What is RL
 - RL doesn't really require a data set, it works on reward hypothesis and tries to achieve the maximum reward instead of learning it using previously played data.
 - When there is a delayed feedback (which works very well for RL, since other types of supervised learning can't)

 - traditional RL performs way better in controlled environments than traditional RL.

 - Read more about AlphaZero and StockFish fights.

 - AlphaZero can play multiple games (using very minimal code changes,[probably just changing the rules]).


### Fundamentals of RL

 - policy - defines the learning agent's way of behaving at a given time.

 - model - the agent's representation of the environment.

 - History - the sequence of observations, actions and rewards.

 - Reward signal - defines the goal of a RL problem.

 - State - a function of history.

 - Information State - contains all the useful informatio from history.

 - State Value - total reward at a particular time.

 - Action Value - the total reward an agent can expect to accumulate if it takes that action.

 - Transitions - predicts the next state.


### K-Armed Bandits & maths

 - A policy where we keep exploitation the first state's maximum value.
 
 - A place where you check everything everytime is called as exploration.

 - the gradient of these two is called as epsilon greedy exploration.

$$
Q_t(a) = \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t}
= \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}}
$$

$$
Q_{n+1} = \frac{1}{n} \sum_{i=1}^{n} R_i
$$

$$
= \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right)
$$

$$
= \frac{1}{n} \left( R_n + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_i \right)
$$

$$
= \frac{1}{n} \left( R_n + (n-1) Q_n \right)
$$

$$
= \frac{1}{n} \left( R_n + n Q_n - Q_n \right)
$$

$$
= Q_n + \frac{1}{n} \left( R_n - Q_n \right)
$$


### Markov Property

 - The future state is independent of the past and depends only on the present state.

 - it could be said it's a tuple containing a finite set of states and their transition probabilities.

### State transition matrix
 - convert each part of the equation into a matrix.


### Markov Reward process

 - MRP are basically with values.

 - Gamma is the discount factor, it ranges from 0 to 1.

 - Gamma close to 0 gives a myopic evaluation of state and action.

 - Gamma close to 1 gives a far-sighted evaluation of state and action.

### Value function and bellman equation

 - Gives the total expected reward if starting in that state.

 
