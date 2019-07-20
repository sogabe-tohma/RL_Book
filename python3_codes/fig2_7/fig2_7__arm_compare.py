import numpy as np
import matplotlib.pyplot as plt

summation = 0
iterate = 0
Nexp = 500
Npulls = 3000
fig = plt.figure()
fig.patch.set_facecolor('white')
class Bandit:
    def __init__(self, N_arm):
        self.N_arm = N_arm
        self.arm_values = np.random.normal(0,1,self.N_arm)
        self.K = np.zeros(self.N_arm)
        self.est_values = np.zeros(self.N_arm)

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
avg_outcome_eps0a1 = np.zeros(Npulls)
avg_outcome_eps0a5 = np.zeros(Npulls)
avg_outcome_eps0a10 = np.zeros(Npulls)
avg_outcome_eps0a100 = np.zeros(Npulls)
avg_outcome_eps0a10000= np.zeros(Npulls)

for i in range(Nexp):
                bandit = Bandit(1)
                avg_outcome_eps0a1 += experiment(bandit,Npulls,0.05)
                bandit = Bandit(5)
                avg_outcome_eps0a5 += experiment(bandit,Npulls,0.05)
                bandit = Bandit(10)
                avg_outcome_eps0a10 += experiment(bandit,Npulls,0.05)
                bandit = Bandit(100)
                avg_outcome_eps0a100 += experiment(bandit,Npulls,0.05)
                bandit = Bandit(10000)
                avg_outcome_eps0a10000 += experiment(bandit,Npulls,0.05)

avg_outcome_eps0a1 /= np.float(Nexp)
avg_outcome_eps0a5 /= np.float(Nexp)
avg_outcome_eps0a10 /= np.float(Nexp)
avg_outcome_eps0a100 /= np.float(Nexp)
avg_outcome_eps0a10000 /= np.float(Nexp)


plt.plot(avg_outcome_eps0a1,label="Arm-1")
plt.plot(avg_outcome_eps0a5,label="Arm-5")
plt.plot(avg_outcome_eps0a10,label="Arm-10")
plt.plot(avg_outcome_eps0a100,label="Arm-100")
plt.plot(avg_outcome_eps0a10000,label="Arm-10000")
plt.ylim(-0.2,2.8)
plt.show()
