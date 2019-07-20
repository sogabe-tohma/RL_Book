
import numpy as np
import gym
from actor_net import ActorNet
from critic_net import CriticNet
from memory import Memory
from ou import OU


MAX_EPISODES = 400
MAX_EP_STEPS = 1000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 400
L2_REG_SCALE = 0
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64
RENDER_ENV = True
GYM_MONITOR_EN = True
ENV_NAME = 'Pendulum-v0'
ONITOR_DIR = './results/gym_ddpg'
ACTION_BOUND=2
ou=OU()


if __name__ == '__main__':

    env = gym.make(ENV_NAME).env

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    actor = ActorNet(state_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, action_dim)
    critic = CriticNet(state_dim, action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS,
                        HIDDEN2_UNITS, action_dim)
    buff = Memory(BUFFER_SIZE,9)
    step=0
    reward_result=[]

    for i in range(MAX_EPISODES):

        s_t = env.reset()
        s_t=np.reshape(s_t,(1,3))[0]
        total_reward = 0.
        for j in range(MAX_EP_STEPS):
            loss=0
            if RENDER_ENV:
                env.render()
            a_t = actor.predict(s_t, ACTION_BOUND, target=False)
            action=a_t+ou.sample(a_t[0])
            s_t_1, r_t, done, info = env.step(action)
            buff.store(s_t, a_t[0], r_t, np.reshape(s_t_1,(1,3))[0], [done])
            if buff.t_memory > MINIBATCH_SIZE:
                batch = buff.sample(MINIBATCH_SIZE)
                states_t = batch[:,0:3]
                actions = batch[:, 3]
                rewards = batch[:, 4]
                b_s_= batch[:,5:8]
                dones =batch[:,-1]
                y=np.zeros((len(batch), 1))
                a_tgt=actor.predict(b_s_, ACTION_BOUND, target=True)
                Q_tgt = critic.predict(b_s_, a_tgt,target=True)
                for i in range(len(batch)):
                    if dones[i]:
                        y[i] = rewards[i]
                    else:
                        y[i] = rewards[i] + GAMMA*Q_tgt[i]
                actions=actions[:,np.newaxis]
                loss += critic.weight_update(states_t, actions, y)
                a_for_dQ_da=actor.predict(states_t, ACTION_BOUND, target=False)
                dQ_da = critic.evaluate_action_gradient(states_t,a_for_dQ_da)
                actor.weight_update(states_t, dQ_da, ACTION_BOUND)
                actor.weight_update_target(TAU)
                critic.weight_update_target(TAU)
            s_t = np.reshape(s_t_1,(1,3))[0]
            total_reward += r_t
            step += 1
            if done:
                "Done!"
                break
        reward_result.append(total_reward)
        print("TOTAL REWARD @ " + str(i) +"-th Episode:" + str(total_reward))
        print("Total Step: " + str(step))
        print("")
