import numpy as np
import matplotlib.pyplot as plt

set_color=["r","b","g","c","m","y","k","w","r","b"]
summation = 0
iterate = 0
Nexp = 5000
Npulls = 3000

class Bandit:
    def __init__(self,N_arm):
        self.N_arm = N_arm
        self.arm_values = np.random.normal(0,1, self.N_arm)
        self.K = np.zeros(self.N_arm)
        self.est_values = np.ones(self.N_arm)*0

    def get_reward(self,action):
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward

    def choose_eps_greedy(self,epsilon):
        rand_num = np.random.random()
        if epsilon>rand_num:
            return np.random.randint(self.N_arm)
        else:
            return np.argmax(self.est_values)

    def update_est(self,action,reward):
        self.K[action] += 1
        alpha = 1./self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action])  
        return self.est_values[action]

def experiment(bandit,Npulls,epsilon):
            history = []
            global summation,iterate
            iterate = iterate + 1
            for i in range(Npulls):
                action = bandit.choose_eps_greedy(epsilon)
                R = bandit.get_reward(action)
                bandit.update_est(action,R)
                history.append(R)
            return np.array(history)


summation  = np.zeros(Npulls)
avg_outcome_eps0p0 = np.zeros(Npulls)
avg_outcome_eps0p05 = np.zeros(Npulls)
avg_outcome_eps0p1 = np.zeros(Npulls)
avg_outcome_eps0p4 = np.zeros(Npulls)
avg_outcome_eps0p8 = np.zeros(Npulls)

for i in range(Nexp):
                bandit = Bandit(10)
                avg_outcome_eps0p0 += experiment(bandit,Npulls,0.0)

                bandit = Bandit(10)
                avg_outcome_eps0p05 += experiment(bandit,Npulls,0.05)

                bandit = Bandit(10)
                avg_outcome_eps0p1 += experiment(bandit,Npulls,0.1)

                bandit = Bandit(10)
                avg_outcome_eps0p4 += experiment(bandit,Npulls,0.4)

                bandit = Bandit(10)
                avg_outcome_eps0p8 += experiment(bandit,Npulls,0.8)

avg_outcome_eps0p0 /= np.float(Nexp)
avg_outcome_eps0p05 /= np.float(Nexp)
avg_outcome_eps0p1 /= np.float(Nexp)
avg_outcome_eps0p4 /= np.float(Nexp)
avg_outcome_eps0p8 /= np.float(Nexp)


plt.plot(avg_outcome_eps0p0,label="eps = 0.0", c= set_color[1])
plt.plot(avg_outcome_eps0p05,label="eps = 0.05",c= set_color[2])
plt.plot(avg_outcome_eps0p1,label="eps = 0.1",c= set_color[3])
plt.plot(avg_outcome_eps0p4,label="eps = 0.4",c= set_color[4])
plt.plot(avg_outcome_eps0p8,label="eps = 0.8",c= set_color[5])

plt.ylim(-0.2,1.8)
plt.show()
