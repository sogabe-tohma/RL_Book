import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

summation = 0
iterate = 0
Nexp = 3
Npulls = 150


fig = plt.figure(figsize=(4,9))
fig.patch.set_facecolor('white')

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
            set_color=["r","b","g","c","m","y","k","w","r","b"]
            for i in range(Npulls):
                action = bandit.choose_eps_greedy(epsilon)
                R = bandit.get_reward(action)
                bandit.update_est(action,R)
                history.append(R)
                plt.subplot(311)
                plt.scatter(i,bandit.update_est(action,R), c=set_color[action],alpha = 1,s =20)
                plt.pause(0.001)


                plt.subplot(312)
                plt.scatter(i,  bandit.get_reward(action), c=set_color[action],alpha = 1,s =20)

                plt.subplot(313)
                summation[i] = summation[i] + history[i]
                def sets_color(l):
                    if l < 0:
                        return "r"  # red
                    else:
                        return "b"  # blue
                plt.scatter(Npulls*(iterate-1)+i, summation[i] / (15.0), c= sets_color(summation[i]),alpha = 0.8,s =20)

                plt.pause(0.001)

            return np.array(history)

summation  = np.zeros(Npulls)
avg_outcome_eps0p8 = np.zeros(Npulls)

for i in range(Nexp):
                bandit = Bandit(10)

                plt.clf()
                avg_outcome_eps0p8 += experiment(bandit,Npulls,0.8)

plt.show()
avg_outcome_eps0p8 /= np.float(Nexp)

fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(avg_outcome_eps0p8,label="eps = 0.8")
plt.legend()
plt.show()
