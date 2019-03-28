# -*- coding: utf-8 -*-

#%% Packages

import numpy as np
#import scipy.stats as spstats
#from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#import seaborn as sns
import itertools
#from scipy.special import erf


  
#%% Approx \nabla \hat{f}
  
  
mu_1 = np.array([-1., 0.])
mu_2 = np.array([1., 0.])
sigma2 =  0.3 # 0.5
s2 = 0.3 # 1.0
nb_bases = 20
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

Lx_pois = np.zeros((meshsize, meshsize, nb_bases**2))
Ly_pois = np.zeros((meshsize, meshsize, nb_bases**2))
dxpois = np.zeros((meshsize, meshsize, nb_bases**2))
dypois = np.zeros((meshsize, meshsize, nb_bases**2))
for k, mu in enumerate(mu_vec):
  dxp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  dyp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  dxpois[:, :, k] = dxp
  dypois[:, :, k] = dyp
  dxdxpois = -(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  dydypois = -(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  
  dxU_dxpois = dxU_mesh * dxp
  dyU_dypois = dyU_mesh * dyp
  
  Lx_pois[:, :, k] = - dxU_dxpois + dxdxpois
  Ly_pois[:, :, k] = - dyU_dypois + dydypois
  
Lx_pois_reshaped = np.reshape(Lx_pois, (meshsize**2, nb_bases**2))    
Ly_pois_reshaped = np.reshape(Ly_pois, (meshsize**2, nb_bases**2))    

L_pois = np.hstack((Lx_pois_reshaped, Ly_pois_reshaped))

def f(y, x):
#  return x + y**3 + np.sin(x) + np.cos(y)
  return x + y
        
def pi_yx(y, x):
  return pi(np.array([x, y]))
  
def pi(x):
  return 0.5*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2)) / (2*np.pi*sigma2) \
          + 0.5*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2)) / (2*np.pi*sigma2)

pi_mesh = 0.5*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) / (2*np.pi*sigma2) \
          + 0.5*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2)) / (2*np.pi*sigma2)  

#pi_f = integrate.dblquad(lambda y, x: f(y, x)*pi_yx(y, x), -bound_y, bound_y, lambda x: -bound_x, lambda x :bound_x)[0]
#f_tilde_mesh = xv + yv**3 + np.sin(xv) + np.cos(yv) - pi_f
f_tilde_mesh = xv + yv # pi_f = 0
f_tilde_flatten = f_tilde_mesh.flatten()

#coeffs_pois = np.linalg.lstsq(L_pois, - f_tilde_flatten, rcond=None)[0]
coeffs_pois = np.linalg.lstsq(L_pois, - f_tilde_flatten, rcond=10**(-3))[0]

dxpois = np.dot(dxpois, coeffs_pois[:nb_bases**2])
dypois = np.dot(dypois, coeffs_pois[nb_bases**2:])
Lpois = np.dot(Lx_pois, coeffs_pois[:nb_bases**2]) + np.dot(Ly_pois, coeffs_pois[nb_bases**2:])
  
var = (dxpois**2 + dypois**2)*pi_mesh
var = var[:-1, :-1]
area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
var = 2*np.sum(var)*area
print('var: {}'.format(var))

def U(x):
  return -np.log(pi(x))

def dU(x):
  dpi = (x-mu_1)*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2))
  dpi += (x-mu_2)*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2))
  dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
  return - dpi / pi(x)

#%% Try to approximate \tilde{f} in L_2(\pi) using directly \nabla \theta_{zv}^{T} \psi

#Lx_psi = np.zeros((meshsize, meshsize, nb_bases**2))
#Ly_psi = np.zeros((meshsize, meshsize, nb_bases**2))
#dxpsi = np.zeros((meshsize, meshsize, nb_bases**2))
#dypsi = np.zeros((meshsize, meshsize, nb_bases**2))
#for k, mu in enumerate(mu_vec):
#  dxp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dyp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dxpsi[:, :, k] = dxp
#  dypsi[:, :, k] = dyp
#  dxdxpsi = -(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dydypsi = -(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  
#  dxU_dxpsi = dxU_mesh * dxp
#  dyU_dypsi = dyU_mesh * dyp
#  
#  Lx_psi[:, :, k] = - dxU_dxpsi + dxdxpsi
#  Ly_psi[:, :, k] = - dyU_dypsi + dydypsi
#
#area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
#f_tilde_mesh = - Lpois
#
#n_mu_vec = len(mu_vec)
#H_zv = np.zeros((2*n_mu_vec, 2*n_mu_vec))
#for i in np.arange(n_mu_vec):
#  for j in np.arange(n_mu_vec):
#    temp = Lx_psi[:,:,i]*Lx_psi[:,:,j]*pi_mesh
#    temp = temp[:-1, :-1]
#    temp = np.sum(temp)*area
#    H_zv[i,j] = temp
#  for j in np.arange(n_mu_vec, 2*n_mu_vec):
#    temp = Lx_psi[:,:,i]*Ly_psi[:,:,j-n_mu_vec]*pi_mesh
#    temp = temp[:-1, :-1]
#    temp = np.sum(temp)*area
#    H_zv[i,j] = temp
#    H_zv[j,i] = temp
#for i in np.arange(n_mu_vec, 2*n_mu_vec):
#  for j in np.arange(n_mu_vec, 2*n_mu_vec):
#    temp = Ly_psi[:,:,i-n_mu_vec]*Ly_psi[:,:,j-n_mu_vec]*pi_mesh
#    temp = temp[:-1, :-1]
#    temp = np.sum(temp)*area
#    H_zv[i,j] = temp
#
#b_zv = np.zeros(2*n_mu_vec)
#for i in np.arange(n_mu_vec):
#  temp = f_tilde_mesh*Lx_psi[:,:,i]*pi_mesh
#  temp = temp[:-1, :-1]
#  temp = np.sum(temp)*area 
#  b_zv[i] = temp
#for i in np.arange(n_mu_vec, 2*n_mu_vec):
#  temp = f_tilde_mesh*Ly_psi[:,:,i-n_mu_vec]*pi_mesh
#  temp = temp[:-1, :-1]
#  temp = np.sum(temp)*area 
#  b_zv[i] = temp
#  
#res = np.linalg.lstsq(H_zv, b_zv, rcond=10**(-2))[0]
#theta_zv = - res[0]
#
#approx_dxpois_zv = np.dot(dxpsi, theta_zv[:n_mu_vec])
#approx_dypois_zv = np.dot(dypsi, theta_zv[n_mu_vec:])


#%% Basis of functions

p = 20 # 5
s2 = 0.3 # 1.0
mu_vec = [np.array([mu1, mu2]) for mu1, mu2 in itertools.product(
    np.linspace(-bound_x, bound_x, num=p), np.linspace(-bound_y, bound_y, num=p))]

n_mu_vec = len(mu_vec)


psi_mesh = [(2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

psi_mesh_dx = [-(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

psi_mesh_dxdy = [(2*np.pi*s2)**(-1)*s2**(-2)*(xv-mu[0])*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
psi_mesh_dxdydy = [(2*np.pi*s2)**(-1)*s2**(-2)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))* \
                   (1.-(yv-mu[1])**2/s2) for mu in mu_vec]

psi_mesh_dxdx = [(2*np.pi*s2)**(-1)*s2**(-1)*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))*((xv-mu[0])**2/s2-1.) for mu in mu_vec]
psi_mesh_dxdxdx = [(2*np.pi*s2)**(-1)*s2**(-2)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))* \
                   (3.-(xv-mu[0])**2/s2) for mu in mu_vec]

psi_mesh_dy = [-(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

psi_mesh_dydx = [(2*np.pi*s2)**(-1)*s2**(-2)*(yv-mu[1])*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
psi_mesh_dydxdx = [(2*np.pi*s2)**(-1)*s2**(-2)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))* \
                   (1.-(xv-mu[0])**2/s2) for mu in mu_vec]

psi_mesh_dydy = [(2*np.pi*s2)**(-1)*s2**(-1)*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))*((yv-mu[1])**2/s2-1.) for mu in mu_vec]
psi_mesh_dydydy = [(2*np.pi*s2)**(-1)*s2**(-2)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))* \
                   (3.-(yv-mu[1])**2/s2) for mu in mu_vec]

psi_mesh = psi_mesh + psi_mesh_dx + psi_mesh_dy

dx_psi_mesh = psi_mesh_dx + psi_mesh_dxdx + psi_mesh_dydx
dy_psi_mesh = psi_mesh_dy + psi_mesh_dxdy + psi_mesh_dydy

dxdx_psi_mesh = psi_mesh_dxdx + psi_mesh_dxdxdx + psi_mesh_dydxdx
dydy_psi_mesh = psi_mesh_dydy + psi_mesh_dxdydy + psi_mesh_dydydy

Lpsi_mesh = [- dx_psi_mesh[i]*dxU_mesh - dy_psi_mesh[i]*dyU_mesh + dxdx_psi_mesh[i] + dydy_psi_mesh[i] 
             for i in np.arange(3*n_mu_vec)]  

nb_approx_basis = 3*p**2

#plt.pcolormesh(xv, yv, dy_psi_mesh[50])
#plt.colorbar()
#plt.show()

area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])

f_tilde_mesh = - Lpois

H = np.zeros((nb_approx_basis, nb_approx_basis))
for i in np.arange(nb_approx_basis):
  for j in np.arange(i, nb_approx_basis):
    temp = (dx_psi_mesh[i]*dx_psi_mesh[j] + dy_psi_mesh[i]*dy_psi_mesh[j]) * pi_mesh
    temp = temp[:-1, :-1]
    temp = np.sum(temp)*area
    H[i,j] = temp
    H[j,i] = temp


H_zv = np.zeros((nb_approx_basis, nb_approx_basis))
for i in np.arange(nb_approx_basis):
  for j in np.arange(i, nb_approx_basis):
    temp = Lpsi_mesh[i]*Lpsi_mesh[j]*pi_mesh
    temp = temp[:-1, :-1]
    temp = np.sum(temp)*area
    H_zv[i,j] = temp    
    H_zv[j,i] = temp    
    
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

np.savez('Hb_gaus_nabla.npz', H=H, H_zv=H_zv, b=b, b_zv=b_zv)
