# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats as spstats
from scipy import signal
import pickle
from multiprocessing import Pool
import multiprocessing

class potentialRegression:
    """ implementing a potential U = logarithm of the posterior distribution
        given by a Bayesian regression
     - Linear
     - Logistic
     - Probit
    """
    
    varY = 1 # Variance of the linear likelihood
    varTheta = 100 # Variance of the prior Gaussian distribution
    
    def __init__(self,Y,X,typ):
        """ initialisation 
        Args:
            Y: observations
            X: covariates
            typ: type of the regression, Linear, Logistic or Probit
        """
        self.Y = Y
        self.X = X
        self.type = typ  
        self.p, self.d = X.shape
    
    def loglikelihood(self,theta):
        """ loglikelihood of the Bayesian regression
        Args:
            theta: parameter of the state space R^d where the likelihood is
                evaluated
        Returns:
            real value of the likelihood evaluated at theta
        """
        if self.type == "g": # Linear regression
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
        """ gradient of the loglikelihood of the Bayesian regression
        Args:
            theta: parameter of the state space R^d where the gradient of the
                likelihood is evaluated
        Returns:
            R^d vector of the gradient of the likelihood evaluated at theta
        """
        if self.type == "g": # Linear
            temp1 = np.dot(np.dot(np.transpose(self.X), self.X), theta)
            temp2 = np.dot(np.transpose(self.X), self.Y)
            return (1. / self.varY)*(temp2 - temp1)
        elif self.type == "l": # Logistic
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
        """ logarithm of the prior distribution, which is a Gaussian distribution
            of variance varTheta
        Args:
            theta: parameter of R^d where the log prior is evaluated
        Returns:
            real value of the log prior evaluated at theta
        """
        return -(1. / (2*self.varTheta))* np.linalg.norm(theta)**2  \
                - (self.d/2.)*np.log(2*np.pi*self.varTheta)
    
    def gradlogprior(self, theta):
        """ gradient of the logarithm of the prior distribution, which is 
            a Gaussian distribution of variance varTheta
        Args:
            theta: parameter of R^d where the gradient log prior is evaluated
        Returns:
            R^d vector of the gradient of the log prior evaluated at theta
        """
        return -(1. / self.varTheta)*theta
    
    def potential(self, theta):
        """ logarithm of the posterior distribution
        Args:
            theta: parameter of R^d where the log posterior is evaluated
        Returns:
            real value of the log posterior evaluated at theta
        """
        return -self.loglikelihood(theta)-self.logprior(theta)
    
    def gradpotential(self, theta):
        """ gradient of the logarithm of the posterior distribution
        Args:
            theta: parameter of R^d where the gradient log posterior is evaluated
        Returns:
            R^d vector of the gradient log posterior evaluated at theta
        """
        return -self.gradloglikelihood(theta)-self.gradlogprior(theta)

""" Samplers ULA, MALA, RWM """
    
def ULA(step, N, n):
    """ MCMC ULA
    Args:
        step: stepsize of the algorithm
        N: burn-in period
        n: number of samples after the burn-in
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the 
            potential U along the trajectory are stored
    """
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    x = np.random.normal(scale=5.0, size=d) # initial value X_0
    for k in np.arange(N): # burn-in period
        x = x - step * potential.gradpotential(x) \
            + np.sqrt(2*step)*np.random.normal(size=d)
    for k in np.arange(n): # samples
        grad = potential.gradpotential(x)
        traj[k,]=x
        traj_grad[k,]=grad
        x = x - step * grad + np.sqrt(2*step)*np.random.normal(size=d)
    return (traj, traj_grad)

def MALA(step, N, n):
    """ MCMC MALA
    Args:
        step: stepsize of the algorithm
        N: burn-in period
        n: number of samples after the burn-in
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the 
            potential U along the trajectory are stored
    """
    U = potential.potential
    grad_U = potential.gradpotential
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    x = np.random.normal(scale=5.0, size=d)
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
    return (traj, traj_grad)

def RWM(step, N, n):
    """ MCMC RWM
    Args:
        step: stepsize of the algorithm
        N: burn-in period
        n: number of samples after the burn-in
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the 
            potential U along the trajectory are stored
    """
    U = potential.potential
    grad_U = potential.gradpotential # for control variates only
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
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
    """ Computation of the empirical means of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        mean_samples: numpy array of size 2*d, containing the means of \theta
            and \theta^2
        var_samples: numpy array of size 2*d, containing the asymptotic 
            variances of \theta and \theta^2
    """
    n, d = traj.shape
    samples = np.concatenate((traj, np.square(traj)), axis=1)
    mean_samples = np.mean(samples, axis=0)
    temp1 = samples - mean_samples
    # Batch Means and spectral variance estimators Flegal and Jones, 2010 
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
    """ Computation of the control variates estimator based on 1st order
        polynomials, CV1, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        mean_CV1: numpy array of size 2*d, containing CV1 applied with 
            the test functions \theta and \theta^2
        var_CV1: numpy array of size 2*d, containing the asymptotic 
            variances of CV1 applied with the test functions \theta and \theta^2
    """
    n, d = traj.shape
    samples = np.concatenate((traj, np.square(traj)), axis=1)
    covariance = np.cov(np.concatenate((traj, samples), axis=1), rowvar=False)
    paramCV1 = covariance[:d, d:]
    CV1 = samples - np.dot(traj_grad, paramCV1)
    mean_CV1 = np.mean(CV1, axis=0)
    CV1 -= mean_CV1
    var_CV1 = np.empty(2*d)
    for i in np.arange(2*d):
        tp1 = (1./n)*signal.fftconvolve(CV1[:,i], CV1[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_CV1[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_CV1, var_CV1)

def CVpolyTwo(traj, traj_grad):
    """ Computation of the control variates estimator based on 2nd order
        polynomials, CV2, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        mean_CV2: numpy array of size 2*d, containing CV2 applied with 
            the test functions \theta and \theta^2
        var_CV2: numpy array of size 2*d, containing the asymptotic 
            variances of CV2 applied with the test functions \theta and \theta^2
    """
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
        tp1 = (1./n)*signal.fftconvolve(CV2[:,i], CV2[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_CV2[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_CV2, var_CV2)

def ZVpolyOne(traj, traj_grad):
    """ Computation of the zero variance estimator based on 1st order
        polynomials, ZV1, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        mean_ZV1: numpy array of size 2*d, containing ZV1 applied with 
            the test functions \theta and \theta^2
        var_ZV1: numpy array of size 2*d, containing the asymptotic 
            variances of ZV1 applied with the test functions \theta and \theta^2
    """
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
        tp1 = (1./n)*signal.fftconvolve(ZV1[:,i], ZV1[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_ZV1[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_ZV1, var_ZV1)

def ZVpolyTwo(traj, traj_grad):
    """ Computation of the zero variance estimator based on 2nd order
        polynomials, ZV2, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        mean_ZV2: numpy array of size 2*d, containing ZV2 applied with 
            the test functions \theta and \theta^2
        var_ZV2: numpy array of size 2*d, containing the asymptotic 
            variances of ZV2 applied with the test functions \theta and \theta^2
    """
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
        tp1 = (1./n)*signal.fftconvolve(ZV2[:,i], ZV2[::-1,i], mode="same")
        tp1 = tp1[:(int(n/2)+1)]
        tp1 = tp1[::-1]
        gam0 = tp1[0]
        bn = int(n**(1./2))
        wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
        var_ZV2[i]= -gam0+2*np.dot(wn, tp1[:bn])
    return (mean_ZV2, var_ZV2)

# Logistic/Probit regression

# Data for the logistic regression
# Swiss dataset
data = np.loadtxt("data\\swiss.txt")
Y = data[:,-1]
X = data[:,0:-1]
# Normalization of the covariates
X = np.dot(X - np.mean(X, axis=0), np.diag(1./np.std(X, axis=0)))

potential = potentialRegression(Y, X, "l")
d = potential.d

# Data for the probit regression
# vaso dataset

#data = np.loadtxt("data\\vaso.txt")
#Y = data[:,-1]
#X = data[:,0:-1]
#X = np.dot(X - np.mean(X, axis=0), np.diag(1./np.std(X, axis=0)))
#X = np.insert(X, 0, 1, axis=1)
#
#potential = potentialRegression(Y, X, "p")
#d = potential.d

#-----------------

""" step size
10**(-2) ULA
5*10**(-2) MALA - 0.574 optimal scaling
5*10**(-2) RWM - optimal acceptance rate scaling 0.234
"""

N = 10**5 # Burn in period
n = 10**6 # Number of samples
step= 10**(-2) # Step size
nc = 100 # Number of independent MCMC trajectories

def func(intseed):
    """ generic function that runs a MCMC trajectory
    and computes means and variances for the ordinary samples, 
    CV1, ZV1, CV2 and ZV2 """
    
    np.random.seed(intseed) # random seed, different for each independent
                            # MCMC trajectory (nc trajectories)
    traj, traj_grad = ULA(step, N, n)
    
    # to save the results of the trajectory 
    sauv = np.zeros((2*5,2*d))
    sauv[0,:], sauv[1,:] = normalSamples(traj,traj_grad) # Normal samples
    sauv[2,:], sauv[3,:] = CVpolyOne(traj,traj_grad) # CV1
    sauv[4,:], sauv[5,:] = CVpolyTwo(traj, traj_grad) # CV2
    sauv[6,:], sauv[7,:] = ZVpolyOne(traj,traj_grad) # ZV1
    sauv[8,:], sauv[9,:] = ZVpolyTwo(traj, traj_grad) # ZV2
        
    return sauv

inputs_seed = np.arange(nc) # input seeds

# number of cores exploited for the computation of the independent trajectories
# by deault, all available cores on the machine
nbcores = multiprocessing.cpu_count()

if __name__ == '__main__':
    trav = Pool(nbcores)
    res = trav.map(func, inputs_seed)
    
    # Save the result
    with open('log_ula_nc100_N5_n6.pkl', 'wb') as f:
        pickle.dump(res, f)

