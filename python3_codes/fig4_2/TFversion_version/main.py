"""
Dependencies:
tensorflow r1.2
keras 2.2.4
"""
import os
import shutil
import random
import numpy as np
import tensorflow as tf
from collections import deque
from grid_world_one_hot import Grid
from DQN_Net import Policy
grid=Grid(4,3)
policy=Policy(12,4)
grid.draw_board()
t_action=grid.t_action
for _ in range(10000):
    grid.resets()
    state=grid.state
    done=grid.done
    actions=[]
    while not done:
        action=policy.choose_action(state)
        observation=grid.step(action)
        n_state=observation[0]
        reward=observation[1]
        done=observation[2]
        policy.learn_act(state,reward,n_state)
        state=n_state
        actions.append(t_action[action])
    print(_,actions)
