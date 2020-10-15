import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style

class FitRadialLamda():
    def __init__(self,x,y,lamda):
        self.x = x
        self.y = y
        self.lamda = lamda

    def hypothesis(self,a3,a4,a5,x,y):
        lamda_hat = a3*np.cos(a4*np.sqrt(np.square(x)+np.square(y))+a5)
        return lamda_hat

    def mse(self,a3,a4,a5,x,y,lamda):
        m = len(lamda)
        lamda_hat = self.hypothesis(a3,a4,a5,x,y)

        errors = abs((lamda_hat-lamda)**2)

        return (1/(m))*np.sum(errors)
    def grad_w3(self,a3,a4,a5,x,y,lamda):
        y_hat = self.hypothesis(a3,a4,a5,x,y)

        return (y_hat-lamda)*(np.cos(a4*np.sqrt(x**2 + y**2) + a5))

    def grad_w4(self, a3, a4, a5, x, y, lamda):
        y_hat = self.hypothesis(a3, a4, a5, x, y)

        return (y_hat - lamda) * (-a3*np.sqrt(x**2 + y**2)*np.sin(a4*np.sqrt(x**2 + y**2) + a5))

    def grad_w5(self, a3, a4, a5, x, y, lamda):
        y_hat = self.hypothesis(a3, a4, a5, x, y)

        return (y_hat - lamda) *(-a3*np.sin(a4*np.sqrt(x**2 + y**2) + a5))

    def fit(self,epochs,learning_rate):
        m = len(self.x)
        a3 = -1
        a4 = 1
        a5 = 1
        mse_list =[]
        for epoch in range(epochs):
            lambda_hat = self.hypothesis(a3,a4,a5,self.x,self.y)
            errors = lambda_hat -self.lamda
            MSE = self.mse(a3,a4,a5,self.x,self.y,self.lamda)
            mse_list.append(MSE)
            da3,da4,da5 = 0,0,0
            for X,Y,Lamda in zip(self.x,self.y,self.lamda):
                da3 += self.grad_w3(a3,a4,a5,X,Y,Lamda)
                da4 += self.grad_w4(a3,a4,a5,X,Y,Lamda)
                da5 += self.grad_w5(a3,a4,a5,X,Y,Lamda)
            a3 = a3 - learning_rate*da3
            a4 = a4 - learning_rate*da4
            a5 = a5 - learning_rate*da5
            print("epoch :"+str(epoch)+"MSE  :",str(MSE))
        self.mser = sum(mse_list)/epochs
        self.epochs = epochs
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

        return self.a3,self.a4,self.a5

    def getmse(self):

        print('Mean Square Error = %.4f' % self.mser)

dataset = pd.read_csv('Radial_datapoints.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:,-1].values
lamda = dataset.iloc[:,-1].values

radialmodel = FitRadialLamda(x,y,lamda)
a3,a4,a5 = radialmodel.fit(1000,0.00005)
print('a3',a3)
print('a4',a4)
print('a5',a5)


