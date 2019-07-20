import numpy as np

class OU(object):
    """
    **OU (Ornstein-Uhlenbeck)**

    DDPG paper ornstein-uhlenbeck noise parameters are theta=0.15, sigma=0.2
    """

    def __init__(self, mu=0, theta=0.15, sigma=0.2, delta=1.0, initial_noise=0):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.delta = delta
        self.n=initial_noise

    def sample(self, action):
        shape = getattr(self.mu, 'shape', [1, ])
        self.n = self.n + self.theta * (self.mu - self.n)*self.delta + np.sqrt(self.delta)*self.sigma \
            * np.random.randn(*shape)

        return self.n
