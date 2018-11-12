import numpy as np
import cv2
from tqdm import tqdm
import time
import signal

#Neural Network class:
class Neural_Network(object):
    def __init__(self, angles, Lambda=0.0001):
        #Define Hyperparameters
        self.inputLayerSize = 2340
        self.outputLayerSize = 64
        self.hiddenLayerSize = 65
        self.angles = angles
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        #limit = sqrt(6 / (self.inputLayerSize + self.hiddenLayerSize))
        #self.W1 = np.random.uniform(-limit, limit, (self.inputLayerSize, self.hiddenLayerSize))
        
        #limit = sqrt(6 / (self.hiddenLayerSize + self.outputLayerSize))
        #self.W2 = np.random.uniform(-limit, limit, (self.hiddenLayerSize, self.outputLayerSize))
        
        #Regularization Parameter:
        self.Lambda = Lambda
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                                                  (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
    
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
    #self.testJ.append(self.N.costFunction(self.testX, self.testY))
    
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
    
    def train(self, trainX, trainY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY
        
        #Make empty list to store training costs:
        self.J = []
        
        params0 = self.N.getParams()
        
        num_iterations = 5000
        alpha = 1e-2
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08
        
        m0 = np.zeros(len(params0))
        v0 = np.zeros(len(params0))
        t = 0.0
        mt = m0
        vt = v0
        
        for i in tqdm(range(num_iterations)):
            if i > 1500:
                alpha = 1e-3
            t += 1
            grads = self.N.computeGradients(X=trainX, y=trainY)
            mt = beta1*mt + (1-beta1)*grads
            vt = beta2*vt + (1-beta2)*grads**2
            mt_hat = mt/(1-beta1**t)
            vt_hat = vt/(1-beta2**t)
            
            params = self.N.getParams()
            new_params = params - alpha*mt_hat/(np.sqrt(vt_hat)+epsilon)
            self.N.setParams(new_params)
