import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

summation = 0
iterate = 0
Nexp = 5000
Npulls = 3000

class Bandit:
    def __init__(self,N_arm):
        self.N_arm = N_arm

        self.arm_values = np.random.normal(0,1, self.N_arm)
        self.K = np.zeros(self.N_arm)
        self.est_values = np.zeros(self.N_arm)

    def get_reward(self,action):
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward

    def choose_eps_greedy(self):
        rand_num = np.random.random()
        if 0.05 > rand_num:
            return np.random.randint(self.N_arm)
        else:
            return np.argmax(self.est_values)

    def update_est(self,action,reward):
        self.K[action] += 1
        alpha = 1./self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action])  
        return self.est_values[action]

def experiment(bandit,Npulls):
            history = []
            global summation,iterate
            iterate = iterate + 1
            for i in range(Npulls):
                action = bandit.choose_eps_greedy()
                R = bandit.get_reward(action)
                bandit.update_est(action,R)
                history.append(R)
                
            return np.array(history)

