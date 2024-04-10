import gym
import simple_driving.envs as senv
# import pybullet_envs
import numpy as np
from assignment3_functions import q_learning
import math
from collections import defaultdict
import pickle
import torch
import random

import pybullet as p
import pybullet_utils.bullet_client as bc

# p.disconnect()
# # Change the connection mode to DIRECT
# p.connect(p.DIRECT)



gamma = 0.95                
episodes = 10000            
max_episode_length = 200    
epsilon = 0.2               
step_size = 0.1 

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
##########################################################################################################################

# env_ = senv.SimpleDrivingEnv(env)
env_ = senv.SimpleDrivingEnv(isDiscrete=True, renders=True)

# env_simple_driving = SimpleDrivingEnv(env)

state, info = env_.reset()

for i in range(200):
    action = q_learning(env_, gamma, episodes, max_episode_length, epsilon, step_size)
    # action = env_.action_space.sample()
    state, reward, done, info = env_.step(action)
    print(env_.car.car)
    print(env_.goal)
    # print(info)
    # if i % 50 == 0:
    #     print("Step: ", i)
    #     print("Information: ", info)
    # obs = env_.getExtendedObservation()
    # print(obs)
    if done:
        break

env_.close()