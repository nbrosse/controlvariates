# -*- coding: utf-8 -*-

#%% Packages

import numpy as np
#import scipy.stats as spstats
from scipy import signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns
from itertools import cycle

# https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions

#%% mixgaus
  
mu_1 = -1.
mu_2 = 1.
sigma2 = 0.5
bound = 5

#for bound in np.arange(3,6):
  
def pi(x):
  return 0.5*np.exp(-(x-mu_1)**2/(2*sigma2)) / (2*np.pi*sigma2)**(1./2) \
          + 0.5*np.exp(-(x-mu_2)**2/(2*sigma2)) / (2*np.pi*sigma2)**(1./2)
           
def U(x):
  return -np.log(pi(x))

def dU(x):
  dpi = (x-mu_1)*np.exp(-(x-mu_1)**2/(2*sigma2))
  dpi += (x-mu_2)*np.exp(-(x-mu_2)**2/(2*sigma2))
  dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1./2)
  return - dpi / pi(x)

def pif(f):
  tp1 = lambda t: f(t)*pi(t)
  return integrate.quad(tp1, -bound, +bound)[0]

def dpoisson(f):
  pi_f = pif(f)
  tp1 = lambda t: -(f(t) - pi_f)*pi(t)
  return lambda x: pi(x)**(-1)*integrate.quad(tp1, -bound, x)[0]

def pi_dpoisson(f):
  pi_f = pif(f)
  tp1 = lambda t: -(f(t) - pi_f)*pi(t)
  return lambda x: integrate.quad(tp1, -bound, x)[0]

def langevin_asymp_variance(f):
  dpois = dpoisson(f)
  tp2 = lambda t: dpois(t)**2*pi(t)
  return 2*integrate.quad(tp2, -bound, bound)[0]


f = lambda x: x + 0.5*x**3 + 3*np.sin(x)
grid = np.linspace(-bound,bound,num=1000)
pi_vec = np.vectorize(pi)
dpois= dpoisson(f)
pi_dpois= pi_dpoisson(f)
dpois_vec = np.vectorize(dpois)
pi_dpois_vec = np.vectorize(pi_dpois)
var = langevin_asymp_variance(f)

print('--------------------------')
print('bound: {}'.format(bound))
print('--------------------------')

print('asymp var: {}'.format(var))

plt.plot(grid, dpois_vec(grid))
plt.grid(True)
plt.title('dpois')
plt.show()

plt.plot(grid, pi_dpois_vec(grid))
plt.grid(True)
plt.title('pi_x_dpois')
plt.show()

plt.plot(grid, np.multiply(dpois_vec(grid), pi_dpois_vec(grid)))
plt.grid(True)
plt.title('pi_x_dpois**2')
plt.show()

plt.plot(grid, pi_vec(grid))
plt.grid(True)
plt.title('pi')
plt.show()



pi_f = pif(f)
var0 = integrate.quad(lambda t: (f(t) - pi_f)**2*pi(t), -bound, bound)[0]

#%% Basis of functions

bound = 5
p = 5
sigma2 = 1.0
mu_vec = np.linspace(-bound, bound, num=p)

pi_f = pif(f)

psi = [(lambda mu: lambda x: (2*np.pi*sigma2)**(-1./2)*np.exp(-(x - mu)**2 / (2*sigma2)))(mu) for mu in mu_vec]
dpsi = [(lambda mu: lambda x: -(2*np.pi*sigma2)**(-1./2)*(x - mu)*np.exp(-(x - mu)**2 / (2*sigma2)) / sigma2)(mu) for mu in mu_vec]

Lpsi = [(lambda i: (lambda mu: lambda x: - dpsi[i](x)*dU(x) + (2*np.pi*sigma2)**(-1./2)*sigma2**(-1)*
         np.exp(-(x - mu)**2 / (2*sigma2))*((x - mu)**2 / sigma2 - 1.))(mu))(i) for (i, mu) in zip(np.arange(p), mu_vec)]

#for i in np.arange(p):
#  plt.plot(grid, np.vectorize(Lpsi[i])(grid))
#plt.grid(True)
#plt.show()

H = np.zeros((p, p))
for i in np.arange(p):
  for j in np.arange(p):
    H[i,j] = integrate.quad(lambda t: dpsi[i](t)*dpsi[j](t)*pi(t), -bound, bound)[0]

H_L = np.zeros((p, p))
for i in np.arange(p):
  for j in np.arange(p):
    H_L[i,j] = integrate.quad(lambda t: Lpsi[i](t)*Lpsi[j](t)*pi(t), -bound, bound)[0]

b = np.zeros(p)
for i in np.arange(p):
  b[i] = integrate.quad(lambda t: (f(t) - pi_f)*psi[i](t)*pi(t), -bound, bound)[0]

b_L = np.zeros(p)
for i in np.arange(p):
  b_L[i] = integrate.quad(lambda t: (f(t) - pi_f)*Lpsi[i](t)*pi(t), -bound, bound)[0]

cond_nb = np.linalg.cond(H)
eig = np.linalg.eigvalsh(H)
eig_L = np.linalg.eigvalsh(H_L)

theta = np.linalg.solve(H, b)
theta_L = - np.linalg.solve(H_L, b_L)

def approx_dpois(t):
  res = 0.
  for i in np.arange(p):
    res += theta[i] * dpsi[i](t)
  return res

def approx_Lpois(t):
  res = 0.
  for i in np.arange(p):
    res += theta[i] * Lpsi[i](t)
  return res

def approx_dpois_L(t):
  res = 0.
  for i in np.arange(p):
    res += theta_L[i] * dpsi[i](t)
  return res

def approx_Lpois_L(t):
  res = 0.
  for i in np.arange(p):
    res += theta_L[i] * Lpsi[i](t)
  return res

plt.plot(grid, np.vectorize(approx_dpois)(grid), label='approx dpois')
plt.plot(grid, np.vectorize(approx_dpois_L)(grid), label='approx dpois_L')
plt.plot(grid, dpois_vec(grid), label='dpois')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(grid, np.vectorize(approx_Lpois)(grid), label='approx Lpois')
plt.plot(grid, np.vectorize(approx_Lpois_L)(grid), label='approx Lpois_L')
plt.plot(grid, - np.vectorize(f)(grid) + pi_f, label='-f_tilde')
plt.grid(True)
plt.legend()
plt.show()

var_cv = 2*integrate.quad(lambda x: (dpois(x) - approx_dpois(x))**2*pi(x), -bound, bound)[0]
var_zv = 2*integrate.quad(lambda x: (dpois(x) - approx_dpois_L(x))**2*pi(x), -bound, bound)[0]

#%% Samplers and analysis

""" Samplers ULA, MALA, RWM """
    
def ULA(step, N, n, gradU):
  traj = np.zeros(n)
  traj_grad = np.zeros(n)
  x = 0. # np.random.normal(scale=5.0) # initial value X_0
  for k in np.arange(N): # burn-in period
    x = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal()
  for k in np.arange(n): # samples
    traj[k] = x
    traj_grad[k] = gradU(x)
    x = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal()
  return (traj, traj_grad)

def MALA(step, N, n, U, gradU):
  traj = np.zeros(n)
  traj_grad = np.zeros(n)
  x = 0. # np.random.normal(scale=5.0, size=d)
#    accept = 0
  for k in np.arange(N):
    y = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal()
    logratio = -U(y)+U(x) + (1./(4*step))*((y-x+step*gradU(x))**2 \
                                         - (x-y+step*gradU(y))**2)
    if np.log(np.random.uniform())<=logratio:
      x = y
  for k in np.arange(n):
    traj[k] = x
    traj_grad[k] = gradU(x)
    y = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal()
    logratio = -U(y)+U(x)+(1./(4*step))*((y-x+step*gradU(x))**2 \
                                         -(x-y+step*gradU(y))**2)
    if np.log(np.random.uniform())<=logratio:
      x = y
#            accept +=1
#    print(np.float(accept) / n)
  return (traj, traj_grad)

def RWM(step, N, n, U, gradU):
  traj = np.zeros(n)
  traj_grad = np.zeros(n)
  x = 0. # np.random.normal(scale=5.0, size=d)
  for k in np.arange(N):
    y = x + np.sqrt(2*step)*np.random.normal()
    logratio = -U(y)+U(x)
    if np.log(np.random.uniform())<=logratio:
      x = y
  for k in np.arange(n):
    traj[k]=x
    traj_grad[k]=gradU(x)
    y = x + np.sqrt(2*step)*np.random.normal()
    logratio = -U(y)+U(x)
    if np.log(np.random.uniform())<=logratio:
      x = y
  return (traj, traj_grad)

""" Control Variates and estimators for mean, asymptotic variance """

def normalSamples(traj,traj_grad,f):
  n = traj.shape[0]
  samples = np.vectorize(f)(traj)
  mean_samples = np.mean(samples)
  temp1 = samples - mean_samples
  # Batch Means and spectral variance estimators Flegal and Jones, 2010 
  tp1 = (1./n)*signal.fftconvolve(temp1, temp1[::-1], mode="same")
  tp1 = tp1[:(int(n/2)+1)]
  tp1 = tp1[::-1]
  gam0 = tp1[0]
  bn = int(n**(1./2))
  wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
  var_samples= -gam0+2*np.dot(wn, tp1[:bn])
  return np.hstack((mean_samples, var_samples)), tp1
#  return tp1

#def CVpolyOne(traj,traj_grad,f):
#    n = traj.shape[0]
#    samples = np.vectorize(f)(traj)
#    covariance = np.cov(np.concatenate((traj[:,np.newaxis], samples[:,np.newaxis]), axis=1), 
#                        rowvar=False)
#    paramCV1 = covariance[0, 1]
#    CV1 = samples - paramCV1*traj_grad
#    mean_CV1 = np.mean(CV1)
#    CV1 -= mean_CV1
#    tp1 = (1./n)*signal.fftconvolve(CV1, CV1[::-1], mode="same")
#    tp1 = tp1[:(int(n/2)+1)]
#    tp1 = tp1[::-1]
#    gam0 = tp1[0]
#    bn = int(n**(1./2))
#    wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
#    var_CV1 = -gam0+2*np.dot(wn, tp1[:bn])
#    return np.hstack((mean_CV1, var_CV1, paramCV1))
#
#def CVft(traj, traj_grad,f,p):
#    n = traj.shape[0]
#    samples = np.vectorize(f)(traj)
#    poisson = np.zeros((n,4*p+2))
#    poisson[:,0] = traj
#    poisson[:,1] = np.power(traj,2)
#    for i in np.arange(1,p+1):
#        poisson[:,i+1] = np.multiply(traj, np.cos(i*traj))
#        poisson[:,i+1+p] = np.multiply(traj, np.sin(i*traj))
#        poisson[:,i+1+2*p] = np.multiply(np.power(traj,2), np.cos(i*traj))
#        poisson[:,i+1+3*p] = np.multiply(np.power(traj,2), np.sin(i*traj))
#    Lpoisson = np.zeros((n,4*p+2))
#    Lpoisson[:,0] = - traj_grad
#    Lpoisson[:,1] = - 2*traj_grad*traj + 2
#    for i in np.arange(1,p+1):        
#        Lpoisson[:,i+1] = - traj_grad * (np.cos(i*traj) - i*traj*np.sin(i*traj)) \
#                        - 2*i*np.sin(i*traj) - i**2*traj*np.cos(i*traj)
#        Lpoisson[:,i+1+p] = - traj_grad * (np.sin(i*traj) + i*traj*np.cos(i*traj)) \
#                        + 2*i*np.cos(i*traj) - i**2*traj*np.sin(i*traj)
#        Lpoisson[:,i+1+2*p] = - traj_grad * (2*traj*np.cos(i*traj) - i*np.power(traj,2)*np.sin(i*traj)) \
#                        + 2*np.cos(i*traj) - 4*i*traj*np.sin(i*traj) - i**2*np.power(traj,2)*np.cos(i*traj)   
#        Lpoisson[:,i+1+3*p] = - traj_grad * (2*traj*np.sin(i*traj) + i*np.power(traj,2)*np.cos(i*traj)) \
#                        + 2*np.sin(i*traj) + 4*i*traj*np.cos(i*traj) - i**2*np.power(traj,2)*np.sin(i*traj)                   
#    cov1 = np.cov(np.concatenate((poisson, -Lpoisson), axis=1), rowvar=False)
#    pbase = poisson.shape[1]
#    A = np.linalg.inv(cov1[0:pbase, pbase:])
#    cov2 = np.cov(np.concatenate((poisson, samples[:,np.newaxis]),axis=1), rowvar=False)
#    B = cov2[0:pbase, pbase:]
#    paramCV = np.dot(A,B)
#    CV = samples + np.dot(Lpoisson, paramCV).flatten()
#    mean_CV = np.mean(CV)
#    CV -= mean_CV
#    tp1 = (1./n)*signal.fftconvolve(CV, CV[::-1], mode="same")
#    tp1 = tp1[:(int(n/2)+1)]
#    tp1 = tp1[::-1]
#    gam0 = tp1[0]
#    bn = int(n**(1./2))
#    wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
#    var_CV = -gam0+2*np.dot(wn, tp1[:bn])
#    return np.hstack((mean_CV, var_CV, paramCV.flatten()))
#
#
#def CVpolyp(traj, traj_grad,f,p):
#    n = traj.shape[0]
#    samples = np.vectorize(f)(traj)
#    poisson = np.zeros((n,p))
#    for i in np.arange(p):
#        poisson[:,i] = np.power(traj, i+1)
#    Lpoisson = np.zeros((n,p))
#    for i in np.arange(p):        
#        Lpoisson[:,i] = (i+1)*i*np.power(traj, i-1) \
#                        - (i+1)*np.multiply(np.power(traj, i), traj_grad)
#    cov1 = np.cov(np.concatenate((poisson, -Lpoisson), axis=1), rowvar=False)
#    A = np.linalg.inv(cov1[0:p, p:])
#    cov2 = np.cov(np.concatenate((poisson, samples[:,np.newaxis]),axis=1), rowvar=False)
#    B = cov2[0:p, p:]
#    paramCV = np.dot(A,B)
#    CV = samples + np.dot(Lpoisson, paramCV).flatten()
#    mean_CV = np.mean(CV)
#    CV -= mean_CV
#    tp1 = (1./n)*signal.fftconvolve(CV, CV[::-1], mode="same")
#    tp1 = tp1[:(int(n/2)+1)]
#    tp1 = tp1[::-1]
#    gam0 = tp1[0]
#    bn = int(n**(1./2))
#    wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
#    var_CV = -gam0+2*np.dot(wn, tp1[:bn])
#    return np.hstack((mean_CV, var_CV, paramCV.flatten()))
#
#def ZVpolyOne(traj, traj_grad,f):
#    n = traj.shape[0]
#    samples = np.vectorize(f)(traj)
#    cov1 = np.cov(traj_grad)
#    A = cov1**(-1)
#    covariance = np.cov(np.concatenate((-traj_grad[:,np.newaxis], samples[:,np.newaxis]), axis=1), 
#                        rowvar=False)
#    paramZV1 = -covariance[0, 1]*A
#    ZV1 = samples - paramZV1*traj_grad
#    mean_ZV1 = np.mean(ZV1)
#    ZV1 -= mean_ZV1
#    tp1 = (1./n)*signal.fftconvolve(ZV1, ZV1[::-1], mode="same")
#    tp1 = tp1[:(int(n/2)+1)]
#    tp1 = tp1[::-1]
#    gam0 = tp1[0]
#    bn = int(n**(1./2))
#    wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
#    var_ZV1= -gam0+2*np.dot(wn, tp1[:bn])
#    return np.hstack((mean_ZV1, var_ZV1, paramZV1))
#
#def ZVft(traj, traj_grad,f,p):
#    n = traj.shape[0]
#    samples = np.vectorize(f)(traj)
#    poisson = np.zeros((n,4*p+2))
#    poisson[:,0] = traj
#    poisson[:,1] = np.power(traj,2)
#    for i in np.arange(1,p+1):
#        poisson[:,i+1] = np.multiply(traj, np.cos(i*traj))
#        poisson[:,i+1+p] = np.multiply(traj, np.sin(i*traj))
#        poisson[:,i+1+2*p] = np.multiply(np.power(traj,2), np.cos(i*traj))
#        poisson[:,i+1+3*p] = np.multiply(np.power(traj,2), np.sin(i*traj))
#    Lpoisson = np.zeros((n,4*p+2))
#    Lpoisson[:,0] = - traj_grad
#    Lpoisson[:,1] = - 2*traj_grad*traj + 2
#    for i in np.arange(1,p+1):        
#        Lpoisson[:,i+1] = - traj_grad * (np.cos(i*traj) - i*traj*np.sin(i*traj)) \
#                        - 2*i*np.sin(i*traj) - i**2*traj*np.cos(i*traj)
#        Lpoisson[:,i+1+p] = - traj_grad * (np.sin(i*traj) + i*traj*np.cos(i*traj)) \
#                        + 2*i*np.cos(i*traj) - i**2*traj*np.sin(i*traj)
#        Lpoisson[:,i+1+2*p] = - traj_grad * (2*traj*np.cos(i*traj) - i*np.power(traj,2)*np.sin(i*traj)) \
#                        + 2*np.cos(i*traj) - 4*i*traj*np.sin(i*traj) - i**2*np.power(traj,2)*np.cos(i*traj)   
#        Lpoisson[:,i+1+3*p] = - traj_grad * (2*traj*np.sin(i*traj) + i*np.power(traj,2)*np.cos(i*traj)) \
#                        + 2*np.sin(i*traj) + 4*i*traj*np.cos(i*traj) - i**2*np.power(traj,2)*np.sin(i*traj)    
#    cov1 = np.cov(Lpoisson, rowvar=False)
#    pbase = poisson.shape[1]
#    A = np.linalg.inv(cov1)
#    cov2 = np.cov(np.concatenate((Lpoisson, samples[:,np.newaxis]),axis=1), rowvar=False)
#    B = cov2[0:pbase, pbase:]
#    paramZV = - np.dot(A,B)
#    ZV = samples + np.dot(Lpoisson, paramZV).flatten()
#    mean_ZV = np.mean(ZV)
#    ZV -= mean_ZV
#    tp1 = (1./n)*signal.fftconvolve(ZV, ZV[::-1], mode="same")
#    tp1 = tp1[:(int(n/2)+1)]
#    tp1 = tp1[::-1]
#    gam0 = tp1[0]
#    bn = int(n**(1./2))
#    wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
#    var_ZV = -gam0+2*np.dot(wn, tp1[:bn])
#    return np.hstack((mean_ZV, var_ZV, paramZV.flatten()))
#
#def ZVpolyp(traj, traj_grad,f,p):
#    n = traj.shape[0]
#    samples = np.vectorize(f)(traj)
#    Lpoisson = np.zeros((n,p))
#    for i in np.arange(p):        
#        Lpoisson[:,i] = (i+1)*i*np.power(traj, i-1) \
#                        - (i+1)*np.multiply(np.power(traj, i), traj_grad)
#    cov1 = np.cov(Lpoisson, rowvar=False)
#    A = np.linalg.inv(cov1)
#    cov2 = np.cov(np.concatenate((Lpoisson, samples[:,np.newaxis]),axis=1), rowvar=False)
#    B = cov2[0:p, p:]
#    paramZV = - np.dot(A,B)
#    ZV = samples + np.dot(Lpoisson, paramZV).flatten()
#    mean_ZV = np.mean(ZV)
#    ZV -= mean_ZV
#    tp1 = (1./n)*signal.fftconvolve(ZV, ZV[::-1], mode="same")
#    tp1 = tp1[:(int(n/2)+1)]
#    tp1 = tp1[::-1]
#    gam0 = tp1[0]
#    bn = int(n**(1./2))
#    wn = 1./2 + (1./2)*np.cos((np.pi/bn)*np.arange(bn))
#    var_ZV = -gam0+2*np.dot(wn, tp1[:bn])
#    return np.hstack((mean_ZV, var_ZV, paramZV.flatten()))

#%% Simulations

""" step size
10**(-2) ULA
5*10**(-2) MALA - 0.574 optimal scaling
5*10**(-2) RWM - optimal acceptance rate scaling 0.234
"""

N = 10**5 # Burn in period
n = 10**6 # Number of samples
step= 10**(-2) # Step size good one 1.

traj, traj_grad = ULA(step,N,n,dU)



ns = normalSamples(traj,traj_grad,f)
ns_cv = normalSamples(traj,traj_grad,lambda t: f(t) + approx_Lpois(t))
ns_zv = normalSamples(traj,traj_grad,lambda t: f(t) + approx_Lpois_L(t))

a, a_cv, a_zv = ns[1], ns_cv[1], ns_zv[1]

nb_cor = 100

plt.bar(np.arange(nb_cor), a[:nb_cor], label='without')
#plt.title('corr without vr')
#plt.show()
#plt.title('corr with cv')
#plt.show()
plt.bar(np.arange(nb_cor), a_zv[:nb_cor], label='zv')
plt.bar(np.arange(nb_cor), a_cv[:nb_cor], label='cv')
#plt.title('corr with zv')
plt.legend()
plt.show()


traj, traj_grad = traj_mala, traj_grad_mala

plt.hist(traj, bins=100, density=True)

pft = 2
sauvft = np.zeros((5, 4*pft+2+2))
sauvft[0,:2] = normalSamples(traj,traj_grad,f) # Normal samples
sauvft[1,:3] = CVpolyOne(traj,traj_grad,f) # CV1
sauvft[2,:3] = ZVpolyOne(traj,traj_grad,f) # ZV1
sauvft[3,:] = CVft(traj,traj_grad,f,pft)
sauvft[4,:] = ZVft(traj,traj_grad,f,pft)

grid = np.linspace(-mg.bound,mg.bound,num=1000)
gradU_grid = np.vectorize(gradU)(grid)

# Construction of the basis function on grid, cos, sin
poisson = np.zeros((1000,4*pft+2))
poisson[:,0] = grid
poisson[:,1] = np.power(grid,2)
for i in np.arange(1,pft+1):
    poisson[:,i+1] = np.multiply(grid, np.cos(i*grid))
    poisson[:,i+1+pft] = np.multiply(grid, np.sin(i*grid))
    poisson[:,i+1+2*pft] = np.multiply(np.power(grid,2), np.cos(i*grid))
    poisson[:,i+1+3*pft] = np.multiply(np.power(grid,2), np.sin(i*grid))
Lpoisson = np.zeros((1000,4*pft+2))
Lpoisson[:,0] = - gradU_grid
Lpoisson[:,1] = - 2*gradU_grid*grid + 2
for i in np.arange(1,pft+1):        
    Lpoisson[:,i+1] = - gradU_grid * (np.cos(i*grid) - i*grid*np.sin(i*grid)) \
                    - 2*i*np.sin(i*grid) - i**2*grid*np.cos(i*grid)
    Lpoisson[:,i+1+pft] = - gradU_grid * (np.sin(i*grid) - i*grid*np.cos(i*grid)) \
                    + 2*i*np.cos(i*grid) - i**2*grid*np.sin(i*grid)
    Lpoisson[:,i+1+2*pft] = - gradU_grid * (2*grid*np.cos(i*grid) - i*np.power(grid,2)*np.sin(i*grid)) \
                    + 2*np.cos(i*grid) - 4*i*grid*np.sin(i*grid) - i**2*np.power(grid,2)*np.cos(i*grid)   
    Lpoisson[:,i+1+3*pft] = - gradU_grid * (2*grid*np.sin(i*grid) + i*np.power(grid,2)*np.cos(i*grid)) \
                    + 2*np.sin(i*grid) + 4*i*grid*np.cos(i*grid) - i**2*np.power(grid,2)*np.sin(i*grid) 

# CV test function cos, sin
plt.plot(grid, grid + (Lpoisson @ sauvft[3,2:]).flatten())
plt.grid(True)
plt.show()

# ZV test function cos, sin
plt.plot(grid, grid + (Lpoisson @ sauvft[4,2:]).flatten())
plt.grid(True)
plt.show()

#CV solution poisson, cos, sin
plt.plot(grid, (poisson @ sauvft[3,2:]).flatten())
plt.grid(True)
plt.show()

#ZV solution poisson, cos, sin
plt.plot(grid, (poisson @ sauvft[4,2:]).flatten())
plt.grid(True)
plt.show()

#%% Polynomials only

p = 10
sauv = np.zeros((2*p+1, 2+p))
sauv[0,:2] = normalSamples(traj,traj_grad,f) # Normal samples
sauv[1,:3] = CVpolyOne(traj,traj_grad,f) # CV1
sauv[2,:3] = ZVpolyOne(traj,traj_grad,f) # ZV1
for i in np.arange(2,p+1):
    sauv[2*i-1,:(i+2)] = CVpolyp(traj,traj_grad,f,i)
    sauv[2*i,:(i+2)] = ZVpolyp(traj,traj_grad,f,i)

grid = np.linspace(-mg.bound,mg.bound,num=1000)
gradU_grid = np.vectorize(gradU)(grid)

# Construction of the basis function, polynomials
A = np.array([(i+1)*(i*np.power(grid,i-1) - np.multiply(np.power(grid,i), gradU_grid)) \
              for i in np.arange(p)]).T

# CV test funcion
plt.figure(1)
ax = plt.gca()
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for i, c in zip(np.arange(p), colors):
    v = grid + np.dot(A, sauv[2*i+1,2:]).flatten()
    plt.plot(grid, v, c=c, label=str(i+1))
plt.title('CV test function')
plt.legend()
plt.grid(True)
plt.show()

# ZV test function
plt.figure(2)
ax = plt.gca()
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for i, c in zip(np.arange(p), colors):
    v = grid + np.dot(A, sauv[2*i+2,2:]).flatten()
    plt.plot(grid, v, c=c, label=str(i+1))
plt.title('ZV test function')
plt.legend()
plt.grid(True)
plt.show()

# Poisson solution

B = np.array([np.power(grid,i+1) for i in np.arange(p)]).T

# CV test funcion
plt.figure(3)
ax = plt.gca()
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for i, c in zip(np.arange(p), colors):
    v = np.dot(B, sauv[2*i+1,2:]).flatten()
    plt.plot(grid, v, c=c, label=str(i+1))
plt.title('CV poisson solution')
plt.legend()
plt.grid(True)
plt.show()

# ZV
plt.figure(4)
ax = plt.gca()
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for i, c in zip(np.arange(p), colors):
    v = np.dot(B, sauv[2*i+2,2:]).flatten()
    plt.plot(grid, v, c=c, label=str(i+1))
plt.title('ZV poisson solution')
plt.legend()
plt.grid(True)
plt.show()


