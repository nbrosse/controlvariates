# -*- coding: utf-8 -*-

#%% Packages

import numpy as np
#import scipy.stats as spstats
from scipy import signal
import scipy.integrate as integrate
#from scipy.special import erf
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#import seaborn as sns
import itertools

#from scipy.optimize import minimize

#%% Approx \nabla \hat{f} point by point.

mu_1 = np.array([-1., 0.])
mu_2 = np.array([1., 0.])
sigma2 =  0.3 # 0.5
s2 = 0.3 # 1.0
nb_bases = 30
bound_x = 3
bound_y = 2

meshsize = 400
x = np.linspace(-bound_x, bound_x, meshsize)
y = np.linspace(-bound_y, bound_y, meshsize)
xv, yv = np.meshgrid(x, y)

mu_vec = list(itertools.product(
           np.linspace(-bound_x, bound_x, num=nb_bases),
           np.linspace(-bound_y, bound_y, num=nb_bases)))

pi_mesh = 0.5*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) / (2*np.pi*sigma2) \
        + 0.5*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2)) / (2*np.pi*sigma2)
dxpi_mesh = (xv-mu_1[0])*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) + \
          (xv-mu_2[0])*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2))
dxpi_mesh *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
dypi_mesh = (yv-mu_1[1])*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) + \
            (yv-mu_2[1])*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2))
dypi_mesh *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
dxU_mesh = - np.divide(dxpi_mesh, pi_mesh)
dyU_mesh = - np.divide(dypi_mesh, pi_mesh)

#Lx_pois = np.zeros((meshsize, meshsize, nb_bases**2))
#Ly_pois = np.zeros((meshsize, meshsize, nb_bases**2))
#dxpois_mesh = np.zeros((meshsize, meshsize, nb_bases**2))
#dypois_mesh = np.zeros((meshsize, meshsize, nb_bases**2))
#for k, mu in enumerate(mu_vec):
#  dxp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dyp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dxpois_mesh[:, :, k] = dxp
#  dypois_mesh[:, :, k] = dyp
#  dxdxpois = -(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dydypois = -(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  
#  dxU_dxpois = dxU_mesh * dxp
#  dyU_dypois = dyU_mesh * dyp
#  
#  Lx_pois[:, :, k] = - dxU_dxpois + dxdxpois
#  Ly_pois[:, :, k] = - dyU_dypois + dydypois
#  
#Lx_pois_reshaped = np.reshape(Lx_pois, (meshsize**2, nb_bases**2))    
#Ly_pois_reshaped = np.reshape(Ly_pois, (meshsize**2, nb_bases**2))    
#
#L_pois = np.hstack((Lx_pois_reshaped, Ly_pois_reshaped))

# ----------- pois parametrized 
dx_pois_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
dy_pois_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

L_pois = np.zeros((meshsize, meshsize, nb_bases**2))
dxpois_mesh = np.zeros((meshsize, meshsize, nb_bases**2))
dypois_mesh = np.zeros((meshsize, meshsize, nb_bases**2))
for i, mu in enumerate(mu_vec):
  dxpois_mesh[:, :, i] = dx_pois_mesh[i]
  dypois_mesh[:, :, i] = dy_pois_mesh[i]
  L_pois[:, :, i] = - dx_pois_mesh[i]*dxU_mesh - dy_pois_mesh[i]*dyU_mesh + (2*np.pi*s2)**(-1)*s2**(-1)* \
                     np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))*(((xv-mu[0])**2+(yv-mu[1])**2)/s2 - 2.)
L_pois_reshaped = np.reshape(L_pois, (meshsize**2, nb_bases**2))          
# --------------------------

def f(y, x):
  return x + y**3 + np.sin(x) + np.cos(y)
#  return x + y
        
def pi_yx(y, x):
  return pi(np.array([x, y]))
  
def pi(x):
  return 0.5*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2)) / (2*np.pi*sigma2) \
          + 0.5*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2)) / (2*np.pi*sigma2)

pi_mesh = 0.5*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) / (2*np.pi*sigma2) \
          + 0.5*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2)) / (2*np.pi*sigma2)  

pi_f = integrate.dblquad(lambda y, x: f(y, x)*pi_yx(y, x), -bound_y, bound_y, lambda x: -bound_x, lambda x :bound_x)[0]
f_tilde_mesh = xv + yv**3 + np.sin(xv) + np.cos(yv) - pi_f
#f_tilde_mesh = xv + yv # pi_f = 0
f_tilde_flatten = f_tilde_mesh.flatten()

coeffs_pois = np.linalg.lstsq(L_pois_reshaped, - f_tilde_flatten, rcond=10**(-8))[0]

dxpois = np.dot(dxpois_mesh, coeffs_pois)
dypois = np.dot(dypois_mesh, coeffs_pois)
Lpois = np.dot(L_pois, coeffs_pois)

var = (dxpois**2 + dypois**2)*pi_mesh
var = var[:-1, :-1]
area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
var = 2*np.sum(var)*area
print(f'var: {var}')
           
  

def U(x):
  return -np.log(pi(x))

def dU(x):
  dpi = (x-mu_1)*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2))
  dpi += (x-mu_2)*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2))
  dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
  return - dpi / pi(x)

#%% CV - ZV - Gaussian kernels

p = 10 # 5
s2 = 1.0 # 1.0
mu_vec = [np.array([mu1, mu2]) for mu1, mu2 in itertools.product(
    np.linspace(-bound_x, bound_x, num=p), np.linspace(-bound_y, bound_y, num=p))]
n_mu_vec = len(mu_vec)

# --------------------------
# Gaussian kernels 
# --------------------------

psi_mesh = [(2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

dx_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
dy_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

Lpsi_mesh = [- dx_psi_mesh[i]*dxU_mesh - dy_psi_mesh[i]*dyU_mesh + (2*np.pi*s2)**(-1)*s2**(-1)*
             np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))*(((xv-mu[0])**2+(yv-mu[1])**2)/s2 - 2.) for i, mu in enumerate(mu_vec)]  

nb_approx_basis = p**2

# -----------------------------------------------------------
# -----------------------------------------------------------

area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])

H = np.zeros((nb_approx_basis, nb_approx_basis))
for i in np.arange(nb_approx_basis):
  for j in np.arange(nb_approx_basis):
    temp = (dx_psi_mesh[i]*dx_psi_mesh[j] + dy_psi_mesh[i]*dy_psi_mesh[j]) * pi_mesh
    temp = temp[:-1, :-1]
    temp = np.sum(temp)*area
    H[i,j] = temp

H_zv = np.zeros((nb_approx_basis, nb_approx_basis))
for i in np.arange(nb_approx_basis):
  for j in np.arange(nb_approx_basis):
    temp = Lpsi_mesh[i]*Lpsi_mesh[j]*pi_mesh
    temp = temp[:-1, :-1]
    temp = np.sum(temp)*area
    H_zv[i,j] = temp    
    
b = np.zeros(nb_approx_basis)
for i in np.arange(nb_approx_basis):
  temp = f_tilde_mesh*psi_mesh[i]*pi_mesh
  temp = temp[:-1, :-1]
  temp = np.sum(temp)*area  
  b[i] = temp

b_zv = np.zeros(nb_approx_basis)
for i in np.arange(nb_approx_basis):
  temp = f_tilde_mesh*Lpsi_mesh[i]*pi_mesh
  temp = temp[:-1, :-1]
  temp = np.sum(temp)*area 
  b_zv[i] = temp

cond_nb = np.linalg.cond(H)
cond_zv = np.linalg.cond(H_zv)
print("cond, cond_zv: {}, {}".format(cond_nb, cond_zv))
eig = np.linalg.eigvalsh(H)
eig_zv = np.linalg.eigvalsh(H_zv)


theta = np.linalg.lstsq(H, b, rcond=10**(-3))[0]
theta_zv = - np.linalg.lstsq(H_zv, b_zv, rcond=10**(-7))[0]
  
approx_dxpois_mesh = np.zeros(dx_psi_mesh[0].shape)
approx_dypois_mesh = np.zeros(dy_psi_mesh[0].shape)
for i in np.arange(nb_approx_basis):
  approx_dxpois_mesh += theta[i] * dx_psi_mesh[i]  
  approx_dypois_mesh += theta[i] * dy_psi_mesh[i]  

approx_Lpois = np.zeros(Lpsi_mesh[0].shape)
approx_Lpois_zv = np.zeros(Lpsi_mesh[0].shape)
for i in np.arange(nb_approx_basis):
  approx_Lpois += theta[i] * Lpsi_mesh[i]  
  approx_Lpois_zv += theta_zv[i] * Lpsi_mesh[i]  

approx_dxpois_zv = np.zeros(dx_psi_mesh[0].shape)
approx_dypois_zv = np.zeros(dy_psi_mesh[0].shape)
for i in np.arange(nb_approx_basis):
  approx_dxpois_zv += theta_zv[i] * dx_psi_mesh[i]  
  approx_dypois_zv += theta_zv[i] * dy_psi_mesh[i]  

#%% Samplers and analysis

""" Samplers ULA, MALA, RWM """
    
def ULA(step, N, n, gradU):
  traj = np.zeros((n, 2))
  traj_grad = np.zeros((n, 2))
  x = np.zeros(2) # np.random.normal(scale=5.0) # initial value X_0
  for k in np.arange(N): # burn-in period
    x = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal(size=2)
  for k in np.arange(n): # samples
    traj[k] = x
    traj_grad[k] = gradU(x)
    x = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal(size=2)
  return (traj, traj_grad)

""" Control Variates and estimators for mean, asymptotic variance """

def analyse_samples(traj, traj_grad, theta=None, gaus=True):
  n = traj.shape[0]
  if theta is None:
    samples = traj[:, 0] + np.power(traj[:, 1], 3) + np.sin(traj[:, 0]) + np.cos(traj[:,1])
  else:
    x = traj[:, 0]
    y = traj[:, 1]
    dpi = (traj-mu_1)*(np.repeat(np.exp(-np.sum((traj-mu_1)**2, axis=1)/(2*sigma2)), 2).reshape(n, 2))
    dpi += (traj-mu_2)*(np.repeat(np.exp(-np.sum((traj-mu_2)**2, axis=1)/(2*sigma2)), 2).reshape(n, 2))
    dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
    
    pi_traj = 0.5*np.exp(-np.sum((traj-mu_1)**2, axis=1)/(2*sigma2)) / (2*np.pi*sigma2) \
          + 0.5*np.exp(-np.sum((traj-mu_2)**2, axis=1)/(2*sigma2)) / (2*np.pi*sigma2)

    dU = - np.divide(dpi, np.repeat(pi_traj, 2).reshape(n, 2))
    if gaus:
      laplacien_psi = [(2*np.pi*s2)**(-1)*s2**(-1)*np.exp(-((x-mu[0])**2+(y-mu[1])**2)/(2*s2))*(((x-mu[0])**2+(y-mu[1])**2)/s2 - 2.) for mu in mu_vec]
      laplacien_psi = np.vstack(tuple(laplacien_psi)).T
      dx_psi= [-(2*np.pi*s2)**(-1)*s2**(-1)*(x-mu[0])*np.exp(-((x-mu[0])**2+(y-mu[1])**2)/(2*s2)) for mu in mu_vec]
      dy_psi= [-(2*np.pi*s2)**(-1)*s2**(-1)*(y-mu[1])*np.exp(-((x-mu[0])**2+(y-mu[1])**2)/(2*s2)) for mu in mu_vec]
      dx_psi = np.vstack(tuple(dx_psi)).T
      dy_psi = np.vstack(tuple(dy_psi)).T
    else:
      dx_psi = np.vstack((np.ones(n), np.zeros(n), y, 2*x, np.zeros(n))).T 
      dy_psi = np.vstack((np.zeros(n), np.ones(n), x, np.zeros(n), 2*y)).T
      laplacien_psi = np.vstack((np.zeros(n), np.zeros(n), np.zeros(n), 
                                 2*np.ones(n), 2*np.ones(n))).T
    laplacien_psi = np.dot(laplacien_psi, theta)
    dx_psi = np.dot(dx_psi, theta)
    dy_psi = np.dot(dy_psi, theta)        

    L_theta_psi = - dU[:,0]*dx_psi - dU[:,1]*dy_psi + laplacien_psi
    
    samples = traj[:, 0] + np.power(traj[:, 1], 3) + np.sin(traj[:, 0]) + np.cos(traj[:,1])
    samples += L_theta_psi

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

N = 10**5 # Burn in period
n = 10**6 # Number of samples
step= 10**(-2) # Step size good one 1.
n_simu = 10

tab_avar = np.zeros(n_simu)
tab_avar_cv = np.zeros(n_simu)
tab_avar_zv = np.zeros(n_simu)

for k in np.arange(n_simu):

  traj, traj_grad = ULA(step,N,n,dU)
  
  ns = analyse_samples(traj,traj_grad)
  ns_cv = analyse_samples(traj,traj_grad,theta=theta)
  ns_zv = analyse_samples(traj,traj_grad,theta=theta_zv)
  
  avar, avar_cv, avar_zv = ns[0][1], ns_cv[0][1], ns_zv[0][1]
  
  tab_avar[k] = avar
  tab_avar_cv[k] = avar_cv
  tab_avar_zv[k] = avar_zv
    
np.save('avar.npy', 
        np.vstack((tab_avar, tab_avar_zv, tab_avar_cv)))
  


