from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU

class Model():

    def __init__(self):
        self.lr=0.001
        self.num_actions=4

    def build_model(self,grid):
        model = Sequential()
        model.add(Dense(grid.size, input_shape=(grid.size,)))
        model.add(PReLU())
        model.add(Dense(grid.size))
        model.add(PReLU())
        model.add(Dense(self.num_actions))
        model.compile(optimizer='adam', loss='mse')

        return model
