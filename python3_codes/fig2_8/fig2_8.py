import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

summation = 0
iterate = 0
iterate_1 = 0
Nexp = 1
Npulls = 150
plt.figure(figsize=(4,9))
class Bandit:
    def __init__(self,N_arm,mean):
        self.N_arm = N_arm
        self.mean_rate = mean
        self.arm_values = np.random.normal(0,1, self.N_arm)
        self.K = np.zeros(self.N_arm)
        self.est_values = np.ones(self.N_arm)*self.mean_rate


    def get_reward(self,action):
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward

    def choose_eps_greedy(self,n):



        action_selected= np.argmax(ucb (bandit.est_values, n, bandit.K))

        print ("arm_average :", ucb (self.est_values, n, self.K),"arm_selected :",action_selected)
        return action_selected

    def update_est(self,action,reward):
        self.K[action] += 1

        alpha = 1./self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action]) 
        return self.est_values[action]

def ucb(mean, n, nj):
    return mean + np.sqrt(2*np.log(n) / (nj + 1e-2))

def run_experiment(bandit,Npulls):
            history = []
            global summation,iterate
            iterate = iterate + 1
            set_color=["r","b","g","c","m","y","k","w","r","b"]
            for i in range(Npulls):
                iterate = iterate + 1
                action = bandit.choose_eps_greedy(iterate)
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
                        return "r"  
                    else:
                        return "b"  
                plt.scatter(Npulls*(iterate-1)+i, summation[i] / (15.0), c= sets_color(summation[i]),alpha = 0.8,s =20)

                plt.pause(0.001)

            return np.array(history)


summation  = np.zeros(Npulls)
avg_outcome= np.zeros(Npulls)


for i in range(Nexp):
    bandit = Bandit(4, 0.0)
    plt.clf()
    avg_outcome += run_experiment(bandit,Npulls)
avg_outcome /= np.float(Nexp)

plt.show()

plt.plot(avg_outcome,label="UCB1",marker='o')
plt.legend()


plt.show()
