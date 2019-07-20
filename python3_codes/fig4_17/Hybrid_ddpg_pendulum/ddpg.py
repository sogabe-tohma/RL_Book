#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# import python packages
import numpy as np
import gym
from actor_net import ActorNet
from critic_net import CriticNet
from ReplayBuffer import ReplayBuffer
import matplotlib.pyplot as plt
MAX_EPISODES = 200
MAX_EP_STEPS = 1000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600
L2_REG_SCALE = 0
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64
RENDER_ENV = True

GYM_MONITOR_EN = True
ENV_NAME = 'Pendulum-v0'
ACTION_BOUND=2
max_time=5000
x= np.linspace(1,MAX_EPISODES,1)
if __name__ == '__main__':
    env = gym.make(ENV_NAME).env
    sa_re=np.zeros((max_time,MAX_EPISODES))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    actor = ActorNet(state_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, action_dim)
    critic = CriticNet(state_dim, action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS,
                        HIDDEN2_UNITS, action_dim)
    buff = ReplayBuffer(BUFFER_SIZE)
    reward_result=np.zeros(MAX_EPISODES)
    for times in range(max_time):

        step=0

        x=np.linspace(1,MAX_EPISODES,MAX_EPISODES)

        for ii in range(MAX_EPISODES):
            s_t = env.reset()
            total_reward = 0.
            count=0
            for j in range(MAX_EP_STEPS):
                loss=0;
                loss2 = 0;
                if RENDER_ENV:
                    env.render()
                a_t = actor.predict(np.reshape(s_t,(1,3)), ACTION_BOUND, target=False)+1./(1.+ii+j)
                s_t_1, r_t, done, info = env.step(a_t[0])
                buff.add(s_t, a_t[0], r_t, s_t_1, done)
                if buff.count() > MINIBATCH_SIZE:
                    batch = buff.getBatch(MINIBATCH_SIZE)
                    states_t = np.asarray([e[0] for e in batch])
                    actions = np.asarray([e[1] for e in batch])
                    rewards = np.asarray([e[2] for e in batch])
                    states_t_1 = np.asarray([e[3] for e in batch])
                    dones = np.asarray([e[4] for e in batch])
                    y=np.zeros((len(batch), action_dim))
                    a_tgt=actor.predict(states_t_1, ACTION_BOUND, target=True)
                    Q_tgt = critic.predict(states_t_1, a_tgt,target=True)

                    for i in range(len(batch)):
                        if dones[i]:
                            y[i] = rewards[i]
                        else:
                            y[i] = rewards[i] + GAMMA*Q_tgt[i]
                    loss += critic.weight_update(states_t, actions, y)
                    a_for_dQ_da=actor.predict(states_t, ACTION_BOUND, target=False)
                    if count==0:
                        dQ_da = critic.evaluate_action_gradient(states_t,a_for_dQ_da)
                        actor.weight_update(states_t, dQ_da, ACTION_BOUND)
                        count=1;
                    else:
                        dL_da = critic.evaluate_action_loss(states_t,a_for_dQ_da,y)
                        actor.weight_update(states_t, dL_da, ACTION_BOUND)
                        count=0;
                    actor.weight_update_target(TAU)
                    critic.weight_update_target(TAU)
                s_t = s_t_1
                total_reward += r_t
                step += 1
                if done:
                    "Done!"
                    break
            reward_result[ii]=(total_reward)
            print("TOTAL REWARD @ " + str(i) +"Episode:" + str(total_reward))
            print("Total Step: " + str(step))
            print("")
            if ii==MAX_EPISODES-1 and j==799:
                plt.plot(x,reward_result)
                plt.xlabel('steps')
                plt.ylabel('rewards')
                plt.hold(True)
    plt.show()
