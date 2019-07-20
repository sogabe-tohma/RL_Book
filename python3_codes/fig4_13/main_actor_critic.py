"""
Dependencies:
tensorflow  1.2
"""
import numpy as np
import tensorflow as tf
from grid_world_env import Grid
from actor import Actor
from critic import Critic

np.random.seed(1)
tf.set_random_seed(2)


MAX_EPISODE = 450
Actor_lr = 0.001
Critic_lr = 0.001

env = Grid()
env.draw_board()
State_dim = 2
Action_dim = 4

sess = tf.Session()

actor = Actor(sess, State_dim=State_dim, Action_dim=Action_dim, lr=Actor_lr)
critic = Critic(sess, State_dim=State_dim, lr=Critic_lr)

sess.run(tf.global_variables_initializer())

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r =0
    total_action=[]
    done= False
    while( not done and t<200):

        a = actor.choose_action(s)

        s_, r, done = env.step(env.t_action[a])
        total_action.append(env.t_action[a])
        if done: r = -200
        td_error = critic.learn(s, -r, s_)
        actor.learn(s, a, td_error)

        s = s_
        track_r+=r
        t += 1
    print("episode:", i_episode, "  tracked actions to attempt goal:",total_action)
