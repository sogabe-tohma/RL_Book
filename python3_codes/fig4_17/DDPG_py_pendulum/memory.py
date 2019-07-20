import numpy as np

class Memory:
    def __init__(self, capacity, dims):
        self.cap=capacity
        self.capacity = np.zeros((capacity, dims))
        self.memory_counter  = 0
        self.t_memory=0

    def store(self, s,r,a,s_,d):
        if self.memory_counter==self.cap:
            self.memory_counter = 0
        transition = np.hstack((s,r,a,s_,d))
        index = self.memory_counter % self.cap
        self.capacity[index, :] = transition
        self.memory_counter += 1
        if self.t_memory<=self.cap-1:
            self.t_memory+=1
        else:
            self.t_memory=self.cap-1

    def sample(self, n):
        if len(self.capacity) > n:
            indices = np.random.choice(self.t_memory, size=n)
        return self.capacity[indices, :]
