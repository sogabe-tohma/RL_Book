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
        self.lambda_ = np.ones(self.N_arm)
        self.mean_rate = 0.0
        self.arm_values = np.random.normal(0,1, self.N_arm)
        self.K = np.zeros(self.N_arm)
        self.est_values = np.ones(self.N_arm)*self.mean_rate
        self.sum_x = np.zeros(self.N_arm) 
        self.tau = np.ones(self.N_arm)
        self.sample = np.zeros(self.N_arm)

    def get_reward(self,action):
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward

    def choose_eps_greedy(self):
        self.sample = np.random.randn() / np.sqrt(self.lambda_) + self.est_values
        return np.argmax(self.sample)

    def update_est(self,action, x):
        self.lambda_[action] += 1
        self.sum_x[action] += x
        self.est_values[action] = self.tau[action]*self.sum_x[action] / self.lambda_[action]
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


