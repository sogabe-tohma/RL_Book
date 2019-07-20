from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
import renom as rm
from renom.optimizer import Adam,Sgd
from renom.utility.initializer import Uniform,GlorotUniform
class build_model(rm.Model):
    
    def __init__(self):
        self.l_1=rm.Dense(49,initializer=Uniform(min=0.0,max=0.3))       
        self.l_2=rm.Dense(49,initializer=Uniform(min=0.0,max=0.3))
        self.l_3=rm.Dense(4,initializer=Uniform(min=0.0,max=0.3))      
    def forward(self,inputs):
        self.inputs=inputs
        h1=rm.leaky_relu(self.l_1(self.inputs))
        h2=rm.leaky_relu(self.l_2(h1))
        h3=self.l_3(h2)
        return h3
    
    def fit(self,inputs,target,epochs):
        for i in range(epochs):
            with self.train():
                value=self.forward(inputs)
            loss = rm.mean_squared_error(value,target)
            loss.grad().update(rm.Adam(lr=0.001))
        return loss