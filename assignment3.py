import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random

import simple_driving.envs


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

env_ = simple_driving.envs.SimpleDrivingEnv(isDiscrete=True, renders=True)

# env_simple_driving = SimpleDrivingEnv(env)

state, info = env_.reset()

for i in range(200):
    action = env_.action_space.sample()
    state, reward, done, info = env_.step(action)
    # print(info)
    # if i % 50 == 0:
    #     print("Step: ", i)
    #     print("Information: ", info)
    obs = env_.getExtendedObservation()
    print(obs)
    if done:
        break

env_.close()