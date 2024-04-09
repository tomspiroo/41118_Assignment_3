import numpy as np
import math 
from collections import defaultdict
import gym

env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)

def epsilon_greedy(env, state, Q, epsilon, episodes, episode):
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = episodes
    sample = np.random.uniform(0, 1)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episode / EPS_DECAY)
    if sample > eps_threshold:
        return np.argmax(Q[state])
    else:
        return env.action_space.sample()

def simulate(env, Q, max_episode_length, epsilon, episodes, episode):
    """Rolls out an episode of actions to be used for learning.

    Args:
        env: gym object.
        Q: state-action value function
        epsilon: control how often you explore random actions versus focusing on
                 high value state and actions
        episodes: maximum number of episodes
        episode: number of episodes played so far

    Returns:
        Dataset of episodes for training the RL agent containing states, actions and rewards.
    """
    D = []
    state = env.reset()                                                     # line 2 - note we don't sample the start state since this is predefined
    done = False
    prev_observation = env.getExtendedObservation()
    for step in range(max_episode_length):                                  # line 3
        action = epsilon_greedy(env, state, Q, epsilon, episodes, episode)  # line 4
        next_state, reward, done, info = env.step(action)                   # line 5
        observation = env.getExtendedObservation()
        #### change reward so that a negative reward is given if the agent falls down a hole #######
        if observation[0] > prev_observation[0] or observation[1] > prev_observation[1]:  #is a greater distance from the goal
            reward = -1.0
        ############################################################################################
        D.append([state, action, reward, next_state])                       # line 7
        state = next_state                                                  # line 8
        prev_observation = observation
        if done:                                                            # if we fall into a hole or reach treasure then end episode
            break
    return D 

def q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size):
    """Main loop of Q-learning algorithm.

    Args:
        env: gym object.
        gamma: discount factor - determines how much to value future actions
        episodes: number of episodes to play out
        max_episode_length: maximum number of steps for episode roll out
        epsilon: control how often you explore random actions versus focusing on
                 high value state and actions
        step_size: learning rate - controls how fast or slow to update Q-values
                   for each iteration.

    Returns:
        Q-function which is used to derive policy.
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))                       # line 2
    total_reward = 0
    for episode in range(episodes):                                             # slightly different to line 3, we just run until maximum episodes played out
        D = simulate(env, Q, max_episode_length, epsilon, episodes, episode)    # line 4
        for data in D:                                                          # data = [state, action, reward, next_state]  (line 5)
            # print(data)
            ####################### update Q value (line 6) #########################
            Q[data[0]][data[1]] = (1 - step_size) * Q[data[0]][data[1]] + step_size * (data[2] +  gamma * np.max(Q[data[3]]))  # line 6
            #########################################################################
            total_reward += data[2]
            # input()
        if episode % 100 == 0:
            print("average total reward per episode batch since episode ", episode, ": ", total_reward/ float(100))
            total_reward = 0
    return Q  # line 9