# -*- coding: utf-8 -*-

#%% Packages

from absl import app
import numpy as np
#import scipy.stats as spstats
#from scipy import signal
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

mu_vec = [(mux, muy) for mux, muy in itertools.product(
           np.linspace(-bound_x, bound_x, num=nb_bases),
           np.linspace(-bound_y, bound_y, num=nb_bases))]

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

#coeffs_pois = np.linalg.lstsq(L_pois, - f_tilde_flatten, rcond=10**(-3))[0]
# ----------- pois
coeffs_pois = np.linalg.lstsq(L_pois_reshaped, - f_tilde_flatten, rcond=10**(-8))[0]

dxpois = np.dot(dxpois_mesh, coeffs_pois)
dypois = np.dot(dypois_mesh, coeffs_pois)
Lpois = np.dot(L_pois, coeffs_pois)
# -----------------


var = (dxpois**2 + dypois**2)*pi_mesh
var = var[:-1, :-1]
area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
var = 2*np.sum(var)*area
print('var: {}'.format(var))

norml1 = np.linalg.norm(coeffs_pois, ord=1) / len(coeffs_pois)  
print('norm coeffs pois: {}'.format(norml1))
err = np.sum(np.absolute(f_tilde_mesh + Lpois)) / meshsize**2
print('err: {}'.format(err))

def U(x):
  return -np.log(pi(x))

def dU(x):
  dpi = (x-mu_1)*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2))
  dpi += (x-mu_2)*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2))
  dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
  return - dpi / pi(x)

#%% CV - ZV - Gaussian kernels

def main(argv):
  p = int(argv[1])
  s2 = float(argv[2])

#  p = 30 
#  s2 = 0.3 
  mu_vec = [np.array([mu1, mu2]) for mu1, mu2 in itertools.product(
      np.linspace(-bound_x, bound_x, num=p), np.linspace(-bound_y, bound_y, num=p))]
#  n_mu_vec = len(mu_vec)
  
  # --------------------------
  # Gaussian kernels 
  # --------------------------
  
  psi_mesh = [(2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
  
  dx_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
  dy_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
  
  Lpsi_mesh = [- dx_psi_mesh[i]*dxU_mesh - dy_psi_mesh[i]*dyU_mesh + (2*np.pi*s2)**(-1)*s2**(-1)*
               np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))*(((xv-mu[0])**2+(yv-mu[1])**2)/s2 - 2.) for i, mu in enumerate(mu_vec)]  
  
  nb_approx_basis = p**2
  
  area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
  
  f_tilde_mesh = - Lpois
  
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
    
  nfile = 'Hb_gaus_p{}_var{}.npz'.format(p, s2)
  
  np.savez(nfile, H=H, H_zv=H_zv, b=b, b_zv=b_zv) # Gaussian kernels
  print('END')

if __name__ == '__main__':
  app.run(main)
