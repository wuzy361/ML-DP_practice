import numpy as np
from matplotlib import pyplot as plt
import random

class linerRegressor:
    def __init__(self,a=0,b=0,learning_rate=0.01):
        self.a = a
        self.b = b
        self.learning_rate = learning_rate
    
    def train(self,x_data,y_data):
        cnt = len(x_data)            
        self.a -= sum(x_data*(self.predict(x_data)-y_data))/cnt
        self.b -= sum(self.predict(x_data)-y_data)/cnt

    def loss(self,x,y):
        return sum((y-self.predict(x))**2)
        
    def predict(self,x):
        return x*self.a + self.b

    def printinfo(self,x_data,y_data):
        print "loss =",linerRegressor1.loss(x_data,y_data)
        print "a = {0},b={1}".format(linerRegressor1.a,linerRegressor1.b)

def data_gen(a,b):
    x = np.random.rand(100).astype(np.float32)
    y = x * a + b
    return x,y

def run(name,times):
    x_data,y_data = data_gen(2.0, 3.0)
    linerRegressor1 = linerRegressor(0,0,0.001)
    
    for x in range(times):
        name.printinfo(x_data,y_data)
        name.train(x_data, y_data)

linerRegressor1 = linerRegressor()

run(linerRegressor1, 100)