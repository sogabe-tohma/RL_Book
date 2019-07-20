import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  bandit_bayes_alg  import Bandit  as Bandit_BA
from  bandit_bayes_alg  import experiment  as experiment_bayes

from  bandit_original_code import Bandit  as  BanditE
from  bandit_original_code import experiment  as  experiment_eps


from  bandit_opt_initial_code import Bandit  as  BanditI
from  bandit_opt_initial_code import experiment  as  experiment_ini


fig = plt.figure()
fig.patch.set_facecolor('white')

summation = 0
iterate = 0
Nexp = 50
Npulls = 3000

class Bandit:
    def __init__(self,N_arm):
        self.N_arm = N_arm

        self.mean_rate = 0
        self.arm_values = np.random.normal(0,1, self.N_arm)
        self.K = np.zeros(self.N_arm)
        self.est_values = np.ones(self.N_arm)*self.mean_rate


    def get_reward(self,action):
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward

    def ucb(self, est_values, n, K):
        return self.est_values + np.sqrt(2*np.log(n) / (self.K + 1e-2))

    def choose_eps_greedy(self,n):
        return np.argmax(bandit.ucb (self.est_values, n, self.K))



    def update_est(self,action,reward):
        self.K[action] += 1

        alpha = 1./self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action]) 
        return self.est_values[action]




def experiment(bandit,Npulls):
            history = []
            global summation,iterate
            iterate =0
            for i in range(Npulls):
                iterate = iterate + 1
                action = bandit.choose_eps_greedy(iterate)
                R = bandit.get_reward(action)
                bandit.update_est(action,R)
                history.append(R)
            return np.array(history)





summation  = np.zeros(Npulls)
avg_outcome_eps0p1 = np.zeros(Npulls)
avg_outcome_eps0p2 = np.zeros(Npulls)
avg_outcome_eps0p3 = np.zeros(Npulls)
avg_outcome_eps0p0 = np.zeros(Npulls)

for i in range(Nexp):

                bandit = Bandit(10)
                avg_outcome_eps0p2 += experiment(bandit,Npulls)

                bandit1 = Bandit_BA(10)
                avg_outcome_eps0p3 += experiment_bayes(bandit1,Npulls)

                bandit1 = BanditE (10)
                avg_outcome_eps0p1 += experiment_eps(bandit1,Npulls)

                bandit1 = BanditI (10)
                avg_outcome_eps0p0 += experiment_ini(bandit1,Npulls)

avg_outcome_eps0p0 /= np.float(Nexp)
avg_outcome_eps0p1 /= np.float(Nexp)
avg_outcome_eps0p2 /= np.float(Nexp)
avg_outcome_eps0p3 /= np.float(Nexp)

plt.plot(avg_outcome_eps0p1,label="e_greddy,eps = 0.05" )
plt.plot(avg_outcome_eps0p0,label="optimal_initial_value" )
plt.plot(avg_outcome_eps0p2,label="UCB1")
plt.plot(avg_outcome_eps0p3,label="Bayes_sampling", color = 'darkmagenta',alpha = 1)

plt.ylim(-0.2,1.8)


plt.show()
