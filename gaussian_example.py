# -*- coding: utf-8 -*-

#%% Objects

""" Production on CMAP
"""

import numpy as np
import scipy.stats as spstats
from scipy import signal
#import sklearn.preprocessing as skpre
#import scipy.integrate as integrate
#import matplotlib.pyplot as plt

#import pickle
#from multiprocessing import Pool
#import multiprocessing

class potentialGaussian:
    
    def __init__(self, Lambda):
        self.Lambda = Lambda
        self.Sigma = np.linalg.inv(self.Lambda)
        self.d = Lambda.shape[0]
        
    def potential(self, theta):
        return np.dot(theta, np.dot(self.Lambda, theta)) / 2.
    
    def gradpotential(self, theta):
        return np.dot(self.Lambda, theta)

class potentialRegression:
    
    varY = 1
    varTheta = 100
    
    def __init__(self,Y,X,typ):
        self.Y = Y
        self.X = X
        self.type = typ  
        self.p, self.d = X.shape
    
    def loglikelihood(self,theta):
        if self.type == "g": # Gaussian
            return -(1. / (2*self.varY))* np.linalg.norm(self.Y-np.dot(self.X,theta))**2 \
                        - (self.d/2.)*np.log(2*np.pi*self.varY)
        elif self.type == "l": # Logistic
            XTheta = np.dot(self.X, theta)
            temp1 = np.dot(self.Y, XTheta)
            temp2 = -np.sum(np.log(1+np.exp(XTheta)))
            return temp1+temp2
        else: # Probit
            cdfXTheta = spstats.norm.cdf(np.dot(self.X, theta))
            cdfMXTheta = spstats.norm.cdf(-np.dot(self.X, theta))
            temp1 = np.dot(self.Y, np.log(cdfXTheta))
            temp2 = np.dot((1 - self.Y), np.log(cdfMXTheta))
            return temp1+temp2
    
    def gradloglikelihood(self, theta):
        if self.type == "g":
            temp1 = np.dot(np.dot(np.transpose(self.X), self.X), theta)
            temp2 = np.dot(np.transpose(self.X), self.Y)
            return (1. / self.varY)*(temp2 - temp1)
        elif self.type == "l":
            temp1 = np.exp(np.dot(self.X, theta))
            temp2 = np.dot(np.transpose(self.X), self.Y)
            temp3 = np.dot(np.transpose(self.X), np.divide(temp1, 1+temp1))
            return temp2 - temp3
        else: # Probit
            XTheta = np.dot(self.X, theta)
            logcdfXTheta = np.log(spstats.norm.cdf(XTheta))
            logcdfMXTheta = np.log(spstats.norm.cdf(-XTheta))
            temp1 = np.multiply(self.Y, np.exp(-0.5*(np.square(XTheta)+np.log(2*np.pi)) \
                                               -logcdfXTheta))
            temp2 = np.multiply((1 -self.Y), np.exp(-0.5*(np.square(XTheta)+np.log(2*np.pi)) \
                                               -logcdfMXTheta))
            return np.dot(np.transpose(self.X), temp1-temp2)
            
        
    def logprior(self, theta):
        return -(1. / (2*self.varTheta))* np.linalg.norm(theta)**2  \
                - (self.d/2.)*np.log(2*np.pi*self.varTheta)
    
    def gradlogprior(self, theta):
        return -(1. / self.varTheta)*theta
    
    def potential(self, theta):
        return -self.loglikelihood(theta)-self.logprior(theta)
    
    def gradpotential(self, theta):
        return -self.gradloglikelihood(theta)-self.gradlogprior(theta)

""" Samplers ULA, MALA, RWM """
    
def ULA(step, N, n):
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
#    x = np.zeros(d)
    x = np.random.normal(scale=5.0, size=d)
    for k in np.arange(N):
        x = x - step * potential.gradpotential(x) \
            + np.sqrt(2*step)*np.random.normal(size=d)
    for k in np.arange(n):
        grad = potential.gradpotential(x)
        traj[k,]=x
        traj_grad[k,]=grad
        x = x - step * grad + np.sqrt(2*step)*np.random.normal(size=d)
    return (traj, traj_grad)

def MALA(step, N, n):
    U = potential.potential
    grad_U = potential.gradpotential
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    x = np.zeros(d)
#    x = np.random.normal(scale=5.0, size=d)
#    accept = 0
    for k in np.arange(N):
        y = x - step * grad_U(x) + np.sqrt(2*step)*np.random.normal(size=d)
        logratio = -U(y)+U(x) + (1./(4*step))*(np.linalg.norm(y-x+step*grad_U(x))**2 \
                      - np.linalg.norm(x-y+step*grad_U(y))**2)
        if np.log(np.random.uniform())<=logratio:
            x = y
    for k in np.arange(n):
        traj[k,]=x
        traj_grad[k,]=grad_U(x)
        y = x - step * grad_U(x) + np.sqrt(2*step)*np.random.normal(size=d)
        logratio = -U(y)+U(x)+(1./(4*step))*(np.linalg.norm(y-x+step*grad_U(x))**2 \
                      -np.linalg.norm(x-y+step*grad_U(y))**2)
        if np.log(np.random.uniform())<=logratio:
            x = y
#            accept += 1
#    print(np.float(accept)/n)
    return (traj, traj_grad)

def RWM(step, N, n):
    U = potential.potential
    grad_U = potential.gradpotential # for control variates only
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
#    x = np.zeros(d)
    x = np.random.normal(scale=5.0, size=d)
    for k in np.arange(N):
        y = x + np.sqrt(2*step)*np.random.normal(size=d)
        logratio = -U(y)+U(x)
        if np.log(np.random.uniform())<=logratio:
            x = y
    for k in np.arange(n):
        traj[k,]=x
        traj_grad[k,]=grad_U(x)
        y = x + np.sqrt(2*step)*np.random.normal(size=d)
        logratio = -U(y)+U(x)
        if np.log(np.random.uniform())<=logratio:
            x = y
    return (traj, traj_grad)

""" Control Variates and estimators for mean, asymptotic variance """

def normalSamples(traj,traj_grad):
    n, d = traj.shape
    samples = np.concatenate((traj, np.square(traj)), axis=1)
    mean_samples = np.mean(samples, axis=0)
    temp1 = samples - mean_samples
    var_samples = np.empty(2*d)
    for i in np.arange(2*d):
        tp1 = (1./n)*signal.fftconvolve(temp1[:,i], temp1[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_samples[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_samples, var_samples)

def CVpolyOne(traj,traj_grad):
    n, d = traj.shape
    samples = np.concatenate((traj, np.square(traj)), axis=1)
    covariance = np.cov(np.concatenate((traj, samples), axis=1), rowvar=False)
    paramCV1 = covariance[:d, d:]
    CV1 = samples - np.dot(traj_grad, paramCV1)
    mean_CV1 = np.mean(CV1, axis=0)
    CV1 -= mean_CV1
    var_CV1 = np.empty(2*d)
    for i in np.arange(2*d):
#        tp1 = (1./n)*np.correlate(CV1[:,i], CV1[:,i], mode="same")
        tp1 = (1./n)*signal.fftconvolve(CV1[:,i], CV1[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_CV1[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_CV1, var_CV1)

def CVpolyTwo(traj, traj_grad):
    n, d = traj.shape
    samples = np.concatenate((traj, np.square(traj)), axis=1)
    poisson = np.zeros((n,int(d*(d+3)/2)))
    poisson[:,np.arange(d)] = traj
    poisson[:,np.arange(d, 2*d)] = np.multiply(traj, traj)
    k = 2*d
    for j in np.arange(d-1):
        for i in np.arange(j+1,d):
            poisson[:,k] = np.multiply(traj[:,i], traj[:,j])
            k=k+1
    Lpoisson = np.zeros((n,int(d*(d+3)/2)))
    Lpoisson[:,np.arange(d)] = - traj_grad
    Lpoisson[:,np.arange(d, 2*d)] = 2*(1. - np.multiply(traj, traj_grad))
    k=2*d
    for j in np.arange(d-1):
        for i in np.arange(j+1,d):
            Lpoisson[:,k] = -np.multiply(traj_grad[:,i], traj[:,j]) \
                    -np.multiply(traj_grad[:,j], traj[:,i])
            k=k+1
    
    cov1 = np.cov(np.concatenate((poisson, -Lpoisson), axis=1), rowvar=False)
    A = np.linalg.inv(cov1[0:int(d*(d+3)/2), int(d*(d+3)/2):d*(d+3)])
    cov2 = np.cov(np.concatenate((poisson, samples),axis=1), rowvar=False)
    B = cov2[0:int(d*(d+3)/2), int(d*(d+3)/2):]
    paramCV2 = np.dot(A,B)
    CV2 = samples + np.dot(Lpoisson, paramCV2)
    mean_CV2 = np.mean(CV2, axis=0)
    CV2 -= mean_CV2
    var_CV2 = np.empty(2*d)
    for i in np.arange(2*d):
#        tp1 = (1./n)*np.correlate(CV2[:,i], CV2[:,i], mode="same")
        tp1 = (1./n)*signal.fftconvolve(CV2[:,i], CV2[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_CV2[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_CV2, var_CV2)

def ZVpolyOne(traj, traj_grad):
    n, d = traj.shape
    samples = np.concatenate((traj, np.square(traj)), axis=1)
    cov1 = np.cov(traj_grad, rowvar=False)
    A = np.linalg.inv(cov1)
    covariance = np.cov(np.concatenate((-traj_grad, samples), axis=1), rowvar=False)
    paramZV1 = -np.dot(A,covariance[:d, d:])
    ZV1 = samples - np.dot(traj_grad, paramZV1)
    mean_ZV1 = np.mean(ZV1, axis=0)
    ZV1 -= mean_ZV1
    var_ZV1 = np.empty(2*d)
    for i in np.arange(2*d):
#        tp1 = (1./n)*np.correlate(ZV1[:,i], ZV1[:,i], mode="same")
        tp1 = (1./n)*signal.fftconvolve(ZV1[:,i], ZV1[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_ZV1[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_ZV1, var_ZV1)

def ZVpolyTwo(traj, traj_grad):
    n, d = traj.shape
    samples = np.concatenate((traj, np.square(traj)), axis=1)
    Lpoisson = np.zeros((n,int(d*(d+3)/2)))
    Lpoisson[:,np.arange(d)] = - traj_grad
    Lpoisson[:,np.arange(d, 2*d)] = 2*(1. - np.multiply(traj, traj_grad))
    k=2*d
    for j in np.arange(d-1):
        for i in np.arange(j+1,d):
            Lpoisson[:,k] = -np.multiply(traj_grad[:,i], traj[:,j]) \
                    -np.multiply(traj_grad[:,j], traj[:,i])
            k=k+1
    
    cov1 = np.cov(Lpoisson, rowvar=False)
    A = np.linalg.inv(cov1)
    cov2 = np.cov(np.concatenate((Lpoisson, samples),axis=1), rowvar=False)
    B = cov2[0:int(d*(d+3)/2), int(d*(d+3)/2):]
    paramZV2 = - np.dot(A,B)
    ZV2 = samples + np.dot(Lpoisson, paramZV2)
    mean_ZV2 = np.mean(ZV2, axis=0)
    ZV2 -= mean_ZV2
    var_ZV2 = np.empty(2*d)
    for i in np.arange(2*d):
#        tp1 = (1./n)*np.correlate(ZV2[:,i], ZV2[:,i], mode="same")
        tp1 = (1./n)*signal.fftconvolve(ZV2[:,i], ZV2[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_ZV2[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_ZV2, var_ZV2)


#%% Gaussian toy example

d = 10
Lambda = np.diag(np.arange(1, 11))

potential = potentialGaussian(Lambda)
d = potential.d

N = 10**5
n = 10**6
#♣ step= 10**(-1) # acceptance ratio 0.5
step= 10**(-2)
traj, traj_grad = MALA(step, N, N)

sauv = np.zeros((2*5,2*d))
sauv[0,:], sauv[1,:] = normalSamples(traj,traj_grad) # Normal samples
sauv[2,:], sauv[3,:] = CVpolyOne(traj,traj_grad) # CV1
sauv[4,:], sauv[5,:] = CVpolyTwo(traj, traj_grad) # CV2
sauv[6,:], sauv[7,:] = ZVpolyOne(traj,traj_grad) # ZV1
sauv[8,:], sauv[9,:] = ZVpolyTwo(traj, traj_grad) # ZV2
