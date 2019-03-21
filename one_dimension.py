# -*- coding: utf-8 -*-

#%% Packages

import numpy as np
#import scipy.stats as spstats
from scipy import signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns

# https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions

# Mixture of gaussians

#%% Study of the truncation boundary
  
#mu_1 = -1.
#mu_2 = 1.
#sigma2 = 0.5
#
#def pi(x):
#  return 0.5*np.exp(-(x-mu_1)**2/(2*sigma2)) / (2*np.pi*sigma2)**(1./2) \
#          + 0.5*np.exp(-(x-mu_2)**2/(2*sigma2)) / (2*np.pi*sigma2)**(1./2)
#           
#def U(x):
#  return -np.log(pi(x))
#
#def dU(x):
#  dpi = (x-mu_1)*np.exp(-(x-mu_1)**2/(2*sigma2))
#  dpi += (x-mu_2)*np.exp(-(x-mu_2)**2/(2*sigma2))
#  dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1./2)
#  return - dpi / pi(x)
#
#f = lambda x: x + 0.5*x**3 + 3*np.sin(x)
#n_grid = 1000
#pi_vec = np.vectorize(pi)
#
##fig = plt.figure(figsize=(10,10))
##plt.rcParams.update({'font.size': 16}) # default 10
##plt.plot(np.linspace(-5,5,num=1000), pi_vec(np.linspace(-5,5,num=1000)))
##plt.grid(True)
##plt.title(r'$\pi$')
##plt.show()
##fig.savefig('pi_1d.pdf', bbox_inches='tight')
#
#
#vec_bound = np.arange(3, 7)
#nb_bound = len(vec_bound)
##bound = 5
#
## Initialization
#tab_var = np.zeros(nb_bound)
#tab_var0 = np.zeros(nb_bound)
#tab_pif = np.zeros(nb_bound)
#tab_dpois = np.zeros((nb_bound, n_grid))
#tab_pi_dpois = np.zeros((nb_bound, n_grid))
#tab_grid = np.zeros((nb_bound, n_grid))
#
#for i, bound in zip(np.arange(nb_bound), vec_bound):
#  
#  def pif(f):
#    tp1 = lambda t: f(t)*pi(t)
#    return integrate.quad(tp1, -bound, +bound)[0]
#  
#  def dpoisson(f):
#    pi_f = pif(f)
#    tp1 = lambda t: -(f(t) - pi_f)*pi(t)
#    return lambda x: pi(x)**(-1)*integrate.quad(tp1, -bound, x)[0]
#  
#  def pi_dpoisson(f):
#    pi_f = pif(f)
#    tp1 = lambda t: -(f(t) - pi_f)*pi(t)
#    return lambda x: integrate.quad(tp1, -bound, x)[0]
#  
#  def langevin_asymp_variance(f):
#    dpois = dpoisson(f)
#    tp2 = lambda t: dpois(t)**2*pi(t)
#    return 2*integrate.quad(tp2, -bound, bound)[0]
#  
#  grid = np.linspace(-bound,bound,num=n_grid)
#  tab_grid[i, :] = grid
#  
#  dpois= dpoisson(f)
#  pi_dpois= pi_dpoisson(f)
#  dpois_vec = np.vectorize(dpois)
#  pi_dpois_vec = np.vectorize(pi_dpois)
#  pi_f = pif(f)
#  var0 = integrate.quad(lambda t: (f(t) - pi_f)**2*pi(t), -bound, bound)[0]
#  var = langevin_asymp_variance(f)
#  
#  tab_pif[i] = pi_f
#  tab_var0[i]= var0
#  tab_var[i] = var
#  tab_dpois[i, :] = dpois_vec(grid)
#  tab_pi_dpois[i, :] = pi_dpois_vec(grid)
#
#
#fig = plt.figure(figsize=(16,16))
#plt.rcParams.update({'font.size': 15}) # default 10
#for k in np.arange(4):
#  plt.subplot(2,2,k+1)
#  plt.plot(tab_grid[k, :], tab_dpois[k, :])
#  plt.title('boundary: {}'.format(vec_bound[k]))
#  plt.grid(True)
#plt.show()
#fig.savefig("dpois_1d.pdf", bbox_inches='tight')
#
#fig = plt.figure(figsize=(16,16))
#plt.rcParams.update({'font.size': 13}) # default 10
#for k in np.arange(4):
#  plt.subplot(2,2,k+1)
#  plt.plot(tab_grid[k, :], tab_pi_dpois[k, :])
#  plt.title('boundary: {}'.format(vec_bound[k]))
#  plt.grid(True)
#plt.show()
#fig.savefig("pi_dpois_1d.pdf", bbox_inches='tight')

#bound_psi = 4
#p = 5
#s2 = 1.0
#mu_vec = np.linspace(-bound_psi, bound_psi, num=p)
#
#psi = [(lambda mu: lambda x: (2*np.pi*s2)**(-1./2)*np.exp(-(x - mu)**2 / (2*s2)))(mu) for mu in mu_vec]
#dpsi = [(lambda mu: lambda x: -(2*np.pi*s2)**(-1./2)*(x - mu)*np.exp(-(x - mu)**2 / (2*s2)) / s2)(mu) for mu in mu_vec]
#
#Lpsi = [(lambda i: (lambda mu: lambda x: - dpsi[i](x)*dU(x) + (2*np.pi*s2)**(-1./2)*s2**(-1)*
#         np.exp(-(x - mu)**2 / (2*s2))*((x - mu)**2 / s2 - 1.))(mu))(i) for (i, mu) in zip(np.arange(p), mu_vec)]
#
##for i in np.arange(p):
##  plt.plot(grid, np.vectorize(Lpsi[i])(grid))
##plt.grid(True)
##plt.show()
#  
#tab_theta = np.zeros((nb_bound, p))
#tab_theta_zv = np.zeros((nb_bound, p))
#
#for k, bound in zip(np.arange(nb_bound), vec_bound):
#  
#  def pif(f):
#    tp1 = lambda t: f(t)*pi(t)
#    return integrate.quad(tp1, -bound, +bound)[0]
#  
#  pi_f = pif(f)
#
#  H = np.zeros((p, p))
#  for i in np.arange(p):
#    for j in np.arange(p):
#      H[i,j] = integrate.quad(lambda t: dpsi[i](t)*dpsi[j](t)*pi(t), -bound, bound)[0]
#  
#  H_zv = np.zeros((p, p))
#  for i in np.arange(p):
#    for j in np.arange(p):
#      H_zv[i,j] = integrate.quad(lambda t: Lpsi[i](t)*Lpsi[j](t)*pi(t), -bound, bound)[0]
#  
#  b = np.zeros(p)
#  for i in np.arange(p):
#    b[i] = integrate.quad(lambda t: (f(t) - pi_f)*psi[i](t)*pi(t), -bound, bound)[0]
#  
#  b_zv = np.zeros(p)
#  for i in np.arange(p):
#    b_zv[i] = integrate.quad(lambda t: (f(t) - pi_f)*Lpsi[i](t)*pi(t), -bound, bound)[0]
#  
#  eig = np.linalg.eigvalsh(H)
#  eig_zv = np.linalg.eigvalsh(H_zv)
#  
#  theta = np.linalg.solve(H, b)
#  theta_zv = - np.linalg.solve(H_zv, b_zv)
#  tab_theta[k, :] = theta
#  tab_theta_zv[k, :] = theta_zv

#%% Bound = 5

mu_1 = -1.
mu_2 = 1.
sigma2 = 0.5
bound = 5

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

f = lambda x: x + 0.5*x**3 + 3*np.sin(x)
n_grid = 1000
grid = np.linspace(-bound,bound,num=n_grid)

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

dpois_vec = np.vectorize(dpoisson(f))
pi_f = pif(f)
var0 = integrate.quad(lambda t: (f(t) - pi_f)**2*pi(t), -bound, bound)[0]
var = langevin_asymp_variance(f)


#%% Basis of functions - study with a varying number of basis functions

#bound_psi = 4
#
#p_vec = np.arange(4, 11)
##p = 5
#n_p = len(p_vec)
#s2 = 1.0
#
## Initialization
#tab_dpois_cv = np.zeros((n_p, n_grid))
#tab_dpois_zv = np.zeros((n_p, n_grid))
#tab_Lpois_cv = np.zeros((n_p, n_grid))
#tab_Lpois_zv = np.zeros((n_p, n_grid))
#
#tab_var_cv = np.zeros(n_p)
#tab_var_zv = np.zeros(n_p)
#tab_var0_cv = np.zeros(n_p)
#tab_var0_zv = np.zeros(n_p)
#
#for k, p in enumerate(p_vec):
#
#  mu_vec = np.linspace(-bound_psi, bound_psi, num=p)
#  
#  psi = [(lambda mu: lambda x: (2*np.pi*s2)**(-1./2)*np.exp(-(x - mu)**2 / (2*s2)))(mu) for mu in mu_vec]
#  dpsi = [(lambda mu: lambda x: -(2*np.pi*s2)**(-1./2)*(x - mu)*np.exp(-(x - mu)**2 / (2*s2)) / s2)(mu) for mu in mu_vec]
#  
#  Lpsi = [(lambda i: (lambda mu: lambda x: - dpsi[i](x)*dU(x) + (2*np.pi*s2)**(-1./2)*s2**(-1)*
#           np.exp(-(x - mu)**2 / (2*s2))*((x - mu)**2 / s2 - 1.))(mu))(i) for (i, mu) in zip(np.arange(p), mu_vec)]
#  
#  #for i in np.arange(p):
#  #  plt.plot(grid, np.vectorize(Lpsi[i])(grid))
#  #plt.grid(True)
#  #plt.show()
#  
#  H = np.zeros((p, p))
#  for i in np.arange(p):
#    for j in np.arange(p):
#      H[i,j] = integrate.quad(lambda t: dpsi[i](t)*dpsi[j](t)*pi(t), -bound, bound)[0]
#  
#  H_zv = np.zeros((p, p))
#  for i in np.arange(p):
#    for j in np.arange(p):
#      H_zv[i,j] = integrate.quad(lambda t: Lpsi[i](t)*Lpsi[j](t)*pi(t), -bound, bound)[0]
#  
#  b = np.zeros(p)
#  for i in np.arange(p):
#    b[i] = integrate.quad(lambda t: (f(t) - pi_f)*psi[i](t)*pi(t), -bound, bound)[0]
#  
#  b_zv = np.zeros(p)
#  for i in np.arange(p):
#    b_zv[i] = integrate.quad(lambda t: (f(t) - pi_f)*Lpsi[i](t)*pi(t), -bound, bound)[0]
#  
#  eig = np.linalg.eigvalsh(H)
#  eig_zv = np.linalg.eigvalsh(H_zv)
#  
#  theta = np.linalg.solve(H, b)
#  theta_zv = - np.linalg.solve(H_zv, b_zv)
#  
#  def approx_dpois(t):
#    res = 0.
#    for i in np.arange(p):
#      res += theta[i] * dpsi[i](t)
#    return res
#  
#  def approx_Lpois(t):
#    res = 0.
#    for i in np.arange(p):
#      res += theta[i] * Lpsi[i](t)
#    return res
#  
#  def approx_dpois_zv(t):
#    res = 0.
#    for i in np.arange(p):
#      res += theta_zv[i] * dpsi[i](t)
#    return res
#  
#  def approx_Lpois_zv(t):
#    res = 0.
#    for i in np.arange(p):
#      res += theta_zv[i] * Lpsi[i](t)
#    return res
#  
#  var_cv = 2*integrate.quad(lambda x: (dpoisson(f)(x) - approx_dpois(x))**2*pi(x), -bound, bound)[0]
#  var_zv = 2*integrate.quad(lambda x: (dpoisson(f)(x) - approx_dpois_zv(x))**2*pi(x), -bound, bound)[0]
#  var0_cv = integrate.quad(lambda x: (f(x) - pi_f + approx_Lpois(x))**2*pi(x), -bound, bound)[0]
#  var0_zv = integrate.quad(lambda x: (f(x) - pi_f + approx_Lpois_zv(x))**2*pi(x), -bound, bound)[0]
#  
#  tab_var_cv[k] = var_cv
#  tab_var_zv[k] = var_zv
#  tab_var0_cv[k] = var0_cv
#  tab_var0_zv[k] = var0_zv
#  
#  tab_dpois_cv[k, :] = np.vectorize(approx_dpois)(grid)
#  tab_dpois_zv[k, :] = np.vectorize(approx_dpois_zv)(grid)
#  tab_Lpois_cv[k, :] = np.vectorize(approx_Lpois)(grid)
#  tab_Lpois_zv[k, :] = np.vectorize(approx_Lpois_zv)(grid)
#
#
##fig = plt.figure(figsize=(16,8*n_p))
##plt.rcParams.update({'font.size': 13}) # default 10
##for k in np.arange(n_p*2):
##  plt.subplot(n_p,2,k+1)
##  if k % 2 == 0:
##    plt.plot(grid, tab_dpois_cv[k//2, :], label=r"$(\theta^{*})^{T} \psi$ '")
##    plt.plot(grid, tab_dpois_zv[k//2, :], label=r"$(\theta^{*}_{zv})^{T} \psi$ '")
##    plt.plot(grid, dpois_vec(grid), label=r"$\hat{f}$ ' ")
##  else:
##    plt.plot(grid, tab_Lpois_cv[k//2, :], label=r"$\mathcal{L} (\theta^{*})^{T} \psi$ ")
##    plt.plot(grid, tab_Lpois_zv[k//2, :], label=r"$\mathcal{L} (\theta^{*}_{zv})^{T} \psi$ ")
##    plt.plot(grid, - np.vectorize(f)(grid) + pi_f, label=r"$-\tilde{f}$")
##  plt.grid(True)
##  plt.legend()
##  plt.title('Number of basis functions: {}'.format(p_vec[k // 2]))
##plt.show()
##fig.savefig("approx_dpois_Lpois_1d.pdf", bbox_inches='tight')
#
#nb_p = 4
#fig = plt.figure(figsize=(16,8*nb_p))
#plt.rcParams.update({'font.size': 13}) # default 10
#for k in np.arange(nb_p*2):
#  plt.subplot(nb_p,2,k+1)
#  k = k + 6
#  if k % 2 == 0:
#    plt.plot(grid, tab_dpois_cv[k//2, :], label=r"$(\theta^{*})^{T} \psi$ '")
#    plt.plot(grid, tab_dpois_zv[k//2, :], label=r"$(\theta^{*}_{zv})^{T} \psi$ '")
#    plt.plot(grid, dpois_vec(grid), label=r"$\hat{f}$ ' ")
#  else:
#    plt.plot(grid, tab_Lpois_cv[k//2, :], label=r"$\mathcal{L} (\theta^{*})^{T} \psi$ ")
#    plt.plot(grid, tab_Lpois_zv[k//2, :], label=r"$\mathcal{L} (\theta^{*}_{zv})^{T} \psi$ ")
#    plt.plot(grid, - np.vectorize(f)(grid) + pi_f, label=r"$-\tilde{f}$")
#  plt.grid(True)
#  plt.legend()
#  plt.title('Number of basis functions: {}'.format(p_vec[k // 2]))
#plt.show()
#fig.savefig("approx_dpois_Lpois_1d_2.pdf", bbox_inches='tight')
#
#
#fig = plt.figure(figsize=(16,8))
#plt.rcParams.update({'font.size': 13}) # default 10
#plt.subplot(1,2,1)
#plt.plot(p_vec, tab_var_cv, 'o--', label=r"$\sigma^{2}_{\infty}(f + \mathcal{L} (\theta^{*})^{T} \psi)$")
#plt.plot(p_vec, tab_var_zv, 'o--', label=r"$\sigma^{2}_{\infty}(f + \mathcal{L} (\theta^{*}_{zv})^{T} \psi)$")
#plt.legend()
#plt.xlabel("Number of basis functions")
#plt.grid(True)
#plt.title("Asymptotic variance")
#plt.subplot(1,2,2)
#plt.plot(p_vec, tab_var0_cv, 'o--', label=r"$\pi((\tilde{f} + \mathcal{L} (\theta^{*})^{T} \psi)^2)$")
#plt.plot(p_vec, tab_var0_zv, 'o--', label=r"$\pi((\tilde{f} + \mathcal{L} (\theta^{*}_{zv})^{T} \psi)^2)$")
#plt.legend()
#plt.xlabel("Number of basis functions")
#plt.grid(True)
#plt.title("Variance")
#plt.show()
#fig.savefig("var_asympt_1d.pdf", bbox_inches='tight')





#plt.plot(grid, np.vectorize(approx_dpois)(grid), label=r"$(\theta^{*})^{T} \psi$ '")
#plt.plot(grid, np.vectorize(approx_dpois_zv)(grid), label=r"$(\theta^{*}_{zv})^{T} \psi$ '")
#plt.plot(grid, dpois_vec(grid), label=r"$\hat{f}$ ' ")
#plt.grid(True)
#plt.legend()
#plt.show()
#
#plt.plot(grid, np.vectorize(approx_Lpois)(grid), label=r"$\mathcal{L} (\theta^{*})^{T} \psi$ ")
#plt.plot(grid, np.vectorize(approx_Lpois_zv)(grid), label=r"$\mathcal{L} (\theta^{*}_{zv})^{T} \psi$ ")
#plt.plot(grid, - np.vectorize(f)(grid) + pi_f, label=r"$-\tilde{f}$")
#plt.grid(True)
#plt.legend()
#plt.show()

#%% p = 4 - Number of basis functions

bound_psi = 4
p = 4
s2 = 1.0

mu_vec = np.linspace(-bound_psi, bound_psi, num=p)

psi = [(lambda mu: lambda x: (2*np.pi*s2)**(-1./2)*np.exp(-(x - mu)**2 / (2*s2)))(mu) for mu in mu_vec]
dpsi = [(lambda mu: lambda x: -(2*np.pi*s2)**(-1./2)*(x - mu)*np.exp(-(x - mu)**2 / (2*s2)) / s2)(mu) for mu in mu_vec]

Lpsi = [(lambda i: (lambda mu: lambda x: - dpsi[i](x)*dU(x) + (2*np.pi*s2)**(-1./2)*s2**(-1)*
         np.exp(-(x - mu)**2 / (2*s2))*((x - mu)**2 / s2 - 1.))(mu))(i) for (i, mu) in zip(np.arange(p), mu_vec)]

H = np.zeros((p, p))
for i in np.arange(p):
  for j in np.arange(p):
    H[i,j] = integrate.quad(lambda t: dpsi[i](t)*dpsi[j](t)*pi(t), -bound, bound)[0]

H_zv = np.zeros((p, p))
for i in np.arange(p):
  for j in np.arange(p):
    H_zv[i,j] = integrate.quad(lambda t: Lpsi[i](t)*Lpsi[j](t)*pi(t), -bound, bound)[0]

b = np.zeros(p)
for i in np.arange(p):
  b[i] = integrate.quad(lambda t: (f(t) - pi_f)*psi[i](t)*pi(t), -bound, bound)[0]

b_zv = np.zeros(p)
for i in np.arange(p):
  b_zv[i] = integrate.quad(lambda t: (f(t) - pi_f)*Lpsi[i](t)*pi(t), -bound, bound)[0]

eig = np.linalg.eigvalsh(H)
eig_zv = np.linalg.eigvalsh(H_zv)

theta = np.linalg.solve(H, b)
theta_zv = - np.linalg.solve(H_zv, b_zv)

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

def approx_dpois_zv(t):
  res = 0.
  for i in np.arange(p):
    res += theta_zv[i] * dpsi[i](t)
  return res

def approx_Lpois_zv(t):
  res = 0.
  for i in np.arange(p):
    res += theta_zv[i] * Lpsi[i](t)
  return res

var_cv = 2*integrate.quad(lambda x: (dpoisson(f)(x) - approx_dpois(x))**2*pi(x), -bound, bound)[0]
var_zv = 2*integrate.quad(lambda x: (dpoisson(f)(x) - approx_dpois_zv(x))**2*pi(x), -bound, bound)[0]
var0_cv = integrate.quad(lambda x: (f(x) - pi_f + approx_Lpois(x))**2*pi(x), -bound, bound)[0]
var0_zv = integrate.quad(lambda x: (f(x) - pi_f + approx_Lpois_zv(x))**2*pi(x), -bound, bound)[0]

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

def samples(traj,traj_grad,f):
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

#%% Simulations

""" step size
10**(-2) ULA
5*10**(-2) MALA - 0.574 optimal scaling
5*10**(-2) RWM - optimal acceptance rate scaling 0.234
"""

n_simu = 10
N = 10**5 # Burn in period
n = 10**6 # Number of samples
step= 10**(-2) # Step size

#tab_avar = np.zeros(n_simu)
#tab_avar_cv = np.zeros(n_simu)
#tab_avar_zv = np.zeros(n_simu)
#
#for k in np.arange(n_simu):
traj, traj_grad = ULA(step,N,n,dU)

ns = samples(traj,traj_grad,f)
ns_cv = samples(traj,traj_grad,lambda t: f(t) + approx_Lpois(t))
ns_zv = samples(traj,traj_grad,lambda t: f(t) + approx_Lpois_zv(t))

avar, avar_cv, avar_zv = ns[0][1], ns_cv[0][1], ns_zv[0][1]

#tab_avar[k] = avar
#tab_avar_cv[k] = avar_cv
#tab_avar_zv[k] = avar_zv


#np.save('avar.npy', 
#        np.vstack((tab_avar, tab_avar_zv, tab_avar_cv)))

a, a_cv, a_zv = ns[1], ns_cv[1], ns_zv[1]

nb_cor = 100

fig = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 13})
plt.bar(np.arange(nb_cor), a[:nb_cor], label=r"$\theta=0$", alpha=0.5)
plt.bar(np.arange(nb_cor), a_zv[:nb_cor], label=r"$\theta=\theta^{*}_{zv}$", alpha=0.5)
plt.bar(np.arange(nb_cor), a_cv[:nb_cor], label=r"$\theta=\theta^{*}$", alpha=0.5)
plt.legend()
plt.title("Autocorrelations")
plt.xlabel(r"$k$")
plt.ylabel(r"$\omega^{h}_{N,n}(k)$")
plt.show()
fig.savefig("autocorrelations_1d.pdf", bbox_inches='tight')


#%% Other basis of functions

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


