# -*- coding: utf-8 -*-

#%% Packages

import numpy as np
#import scipy.stats as spstats
from scipy import signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#import seaborn as sns
import itertools

#from scipy.optimize import minimize

# Mixture of two gaussians

#%% Approx. \nabla\hat{f} - study of the truncation boundaries

#mu_1 = np.array([-1., 0.])
#mu_2 = np.array([1., 0.])
#sigma2 = 0.5
#s2 = 1.0
#nb_bases = 5
#bound_x = 3
#bound_y = 2
#tab_bound_x = np.arange(2.5, 5, step=0.2)
#tab_bound_y = np.arange(1.5, 4, step=0.2)
#tab_var = np.zeros((len(tab_bound_x), len(tab_bound_y)))
#
#
#
#for ix, iy in itertools.product(np.arange(len(tab_bound_x)), np.arange(len(tab_bound_y))):
#    
#  bound_x = tab_bound_x[ix]
#  bound_y = tab_bound_y[iy]
#  
#  print('--------------------------')
#  print('bound_x: {} , bound_y: {}'.format(bound_x, bound_y))
#  print('--------------------------')
#  
#  meshsize = 100
#  x = np.linspace(-bound_x, bound_x, meshsize)
#  y = np.linspace(-bound_y, bound_y, meshsize)
#  xv, yv = np.meshgrid(x, y)
#  
#  mu_vec = [(mux, muy) for mux, muy in itertools.product(
#             np.linspace(-bound_x, bound_x, num=nb_bases),
#             np.linspace(-bound_y, bound_y, num=nb_bases))]
#
#  pi_mesh = 0.5*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) / (2*np.pi*sigma2) \
#          + 0.5*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2)) / (2*np.pi*sigma2)  
#  dxpi_mesh = (xv-mu_1[0])*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) + \
#            (xv-mu_2[0])*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2))
#  dxpi_mesh *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
#  dypi_mesh = (yv-mu_1[1])*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) + \
#              (yv-mu_2[1])*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2))
#  dypi_mesh *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
#  dxU_mesh = - np.divide(dxpi_mesh, pi_mesh)
#  dyU_mesh = - np.divide(dypi_mesh, pi_mesh)
#
#  Lx_pois = np.zeros((meshsize, meshsize, nb_bases**2))
#  Ly_pois = np.zeros((meshsize, meshsize, nb_bases**2))
#  dxpois = np.zeros((meshsize, meshsize, nb_bases**2))
#  dypois = np.zeros((meshsize, meshsize, nb_bases**2))
#  for k, mu in enumerate(mu_vec):
#    dxp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#    dyp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#    dxpois[:, :, k] = dxp
#    dypois[:, :, k] = dyp
#    dxdxpois = -(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#    dydypois = -(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#    
#    dxU_dxpois = dxU_mesh * dxp
#    dyU_dypois = dyU_mesh * dyp
#    
#    Lx_pois[:, :, k] = - dxU_dxpois + dxdxpois
#    Ly_pois[:, :, k] = - dyU_dypois + dydypois
#    
#  Lx_pois = np.reshape(Lx_pois, (meshsize**2, nb_bases**2))    
#  Ly_pois = np.reshape(Ly_pois, (meshsize**2, nb_bases**2))    
#  
#  L_pois = np.hstack((Lx_pois, Ly_pois))
#  
#  def f(y, x):
#    return x + y**3 + np.sin(x) + np.cos(y)
#          
#  def pi_yx(y, x):
#    return pi(np.array([x, y]))
#    
#  def pi(x):
#    return 0.5*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2)) / (2*np.pi*sigma2) \
#            + 0.5*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2)) / (2*np.pi*sigma2)
#  
#  pi_mesh = 0.5*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) / (2*np.pi*sigma2) \
#            + 0.5*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2)) / (2*np.pi*sigma2)  
#  
#  pi_f = integrate.dblquad(lambda y, x: f(y, x)*pi_yx(y, x), -bound_y, bound_y, lambda x: -bound_x, lambda x :bound_x)[0]
#  f_tilde_mesh = xv + yv**3 + np.sin(xv) + np.cos(yv) - pi_f
#  f_tilde_flatten = f_tilde_mesh.flatten()
#  
#  coeffs_pois = np.linalg.lstsq(L_pois, - f_tilde_flatten, rcond=None)[0]
#  
#  dxpois = np.dot(dxpois, coeffs_pois[:nb_bases**2])
#  dypois = np.dot(dypois, coeffs_pois[nb_bases**2:])
#  
##  fig = plt.figure()
##  ax = fig.gca(projection='3d')
##  ax.plot_surface(xv, yv, dypois) # cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
##  ax.plot_surface(xv[10:40, 10:40], yv[10:40, 10:40], dxpois[10:40, 10:40]) # cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
#  
#  #plt.contour(xv, yv, f_tilde_mesh)
#  #plt.colorbar()
#  #plt.show()  
#  
#  
#  plt.contour(xv, yv, dxpois)
#  plt.colorbar()
#  plt.title("dxpois")
#  plt.show()
#  
#  plt.contour(xv, yv, dypois)
#  plt.colorbar()
#  plt.title("dypois")
#  plt.show()
#             
#  
#  var = (dxpois**2 + dypois**2)*pi_mesh
#  var = var[:-1, :-1]
#  area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
#  var = 2*np.sum(var)*area
#  tab_var[ix, iy] = var
#  print('var: {}'.format(var))

#diff_var = (np.roll(tab_var, 1, axis=0) - tab_var)**2 + \
#        (np.roll(tab_var, -1, axis=0) - tab_var)**2 + \
#        (np.roll(tab_var, 1, axis=1) - tab_var)**2 + \
#        (np.roll(tab_var, -1, axis=1) - tab_var)**2 
#
#diff_var = diff_var[1:-1, 1:-1]


  
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

# ----------- dxpois and dypois parametrized independently
#Lx_pois = np.zeros((meshsize, meshsize, nb_bases**2))
#Ly_pois = np.zeros((meshsize, meshsize, nb_bases**2))
#dxpois = np.zeros((meshsize, meshsize, nb_bases**2))
#dypois = np.zeros((meshsize, meshsize, nb_bases**2))
#for k, mu in enumerate(mu_vec):
#  dxp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dyp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
#  dxpois[:, :, k] = dxp
#  dypois[:, :, k] = dyp
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
# --------------------------

# ----------- pois parametrized 
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

dx_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
dy_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

Lpsi_mesh = [- dx_psi_mesh[i]*dxU_mesh - dy_psi_mesh[i]*dyU_mesh + (2*np.pi*s2)**(-1)*s2**(-1)*
             np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))*(((xv-mu[0])**2+(yv-mu[1])**2)/s2 - 2.) for i, mu in enumerate(mu_vec)]  

# --------------------------

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

#  fig = plt.figure()
#  ax = fig.gca(projection='3d')
#  ax.plot_surface(xv, yv, dypois) # cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
#  ax.plot_surface(xv[10:40, 10:40], yv[10:40, 10:40], dxpois[10:40, 10:40]) # cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
  
plt.pcolormesh(xv, yv, f_tilde_mesh)
plt.colorbar()
plt.title(r"$\tilde{f}$")
plt.show()  
  
plt.pcolormesh(xv, yv, -Lpois)
plt.colorbar()
plt.title(r"$-\mathcal{L} \hat{f}$")
plt.show()  


#plt.contour(xv, yv, dxpois)
#plt.colorbar()
#plt.title("dxpois")
#plt.show()

plt.pcolormesh(xv, yv, dxpois)
plt.colorbar()
plt.title("dxpois")
plt.show()

#plt.contour(xv, yv, dypois)
#plt.colorbar()
#plt.title("dypois")
#plt.show()

plt.pcolormesh(xv, yv, dypois)
#plt.contour(xv, yv, pi_mesh)
plt.colorbar()
plt.title("dypois")
plt.show()

fig = plt.figure(figsize=(10,10))
plt.rcParams.update({'font.size': 16}) # default 10
plt.pcolormesh(xv, yv, pi_mesh)
plt.colorbar()
plt.title(r'$\pi$')
plt.show()
#fig.savefig('density_pi_2d.pdf', bbox_inches='tight')
           
  
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



#%% Polynomial basis of functions
  
ones_mesh = np.ones(xv.shape)
zeros_mesh = np.zeros(xv.shape)

psi_mesh = [xv, yv, xv*yv, xv**2, yv**2]
nb_approx_basis = len(psi_mesh)
dx_psi_mesh = [ones_mesh, zeros_mesh, yv, 2*xv, zeros_mesh]
dy_psi_mesh = [zeros_mesh, ones_mesh, xv, zeros_mesh, 2*yv]

laplacien_psi_mesh = [zeros_mesh, zeros_mesh, zeros_mesh, 2.*ones_mesh, 2.*ones_mesh]
Lpsi_mesh = [- dx_psi_mesh[i]*dxU_mesh - dy_psi_mesh[i]*dyU_mesh + laplacien_psi_mesh[i] 
              for i in np.arange(nb_approx_basis)]  

#%% Basis of functions

p = 20 # 5
s2 = 0.3 # 1.0
mu_vec = [np.array([mu1, mu2]) for mu1, mu2 in itertools.product(
    np.linspace(-bound_x, bound_x, num=p), np.linspace(-bound_y, bound_y, num=p))]


#psi = [(lambda mu1, mu2: lambda y, x: (2*np.pi*s2)**(-1)*np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2)))(mu[0], mu[1]) for mu in mu_vec]

psi_mesh = [(2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

#dx_psi = [(lambda mu1, mu2: lambda y, x: -(2*np.pi*s2)**(-1)*s2**(-1)*(x-mu1)*np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2)))(mu[0], mu[1]) for mu in mu_vec]
#dy_psi = [(lambda mu1, mu2: lambda y, x: -(2*np.pi*s2)**(-1)*s2**(-1)*(y-mu2)*np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2)))(mu[0], mu[1]) for mu in mu_vec]

dx_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]
dy_psi_mesh = [-(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

#Lpsi = [(lambda i: (lambda mu1, mu2: lambda y, x: - dx_psi[i](y, x)*dU([x, y])[0] - dy_psi[i](y, x)*dU([x, y])[1] + (2*np.pi*s2)**(-1)*s2**(-1)*
#         np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2))*(((x-mu1)**2+(y-mu2)**2)/s2 - 2.))(mu[0], mu[1]))(i) for (i, mu) in zip(np.arange(p**2), mu_vec)]

Lpsi_mesh = [- dx_psi_mesh[i]*dxU_mesh - dy_psi_mesh[i]*dyU_mesh + (2*np.pi*s2)**(-1)*s2**(-1)*
             np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2))*(((xv-mu[0])**2+(yv-mu[1])**2)/s2 - 2.) for i, mu in enumerate(mu_vec)]  

nb_approx_basis = p**2
  
#for i in np.arange(p**2):
#  plt.contour(xv, yv, psi_mesh[i])
#  plt.show()

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

#np.savez('Hb.npz', H=H, H_zv=H_zv, b=b, b_zv=b_zv)
npzfile = np.load('Hb.npz')
H = npzfile['H']
H_zv = npzfile['H_zv']
b = npzfile['b']
b_zv = npzfile['b_zv']

# ------------- Expensive
  
#H = np.zeros((p**2, p**2))
#for i in np.arange(p**2):
#  for j in np.arange(p**2):
#    H[i,j] = integrate.dblquad(lambda y, x: (dx_psi[i](y, x)*dx_psi[j](y, x)+
#     dy_psi[i](y, x)*dy_psi[j](y, x))*pi_yx(y, x), -bound_y, bound_y, lambda x:-bound_x, lambda x:bound_x)[0]
#
#H_zv = np.zeros((p**2, p**2))
#for i in np.arange(p**2):
#  for j in np.arange(p**2):
#    H_zv[i,j] = integrate.dblquad(lambda y, x: Lpsi[i](y, x)*Lpsi[j](y, x)*pi_yx(y, x), 
#        -bound_y, bound_y, lambda x:-bound_x, lambda x:bound_x)[0]
#
#b = np.zeros(p**2)
#for i in np.arange(p**2):
#  b[i] = integrate.dblquad(lambda y, x: (f(y, x) - pi_f)*psi[i](y, x)*pi_yx(y, x), 
#   -bound_y, bound_y, lambda x:-bound_x, lambda x:bound_x)[0]
#
#b_zv = np.zeros(p**2)
#for i in np.arange(p**2):
#  b_zv[i] = integrate.dblquad(lambda y, x: (f(y, x) - pi_f)*Lpsi[i](y, x)*pi_yx(y, x), 
#      -bound_y, bound_y, lambda x:-bound_x, lambda x:bound_x)[0]
#
cond_nb = np.linalg.cond(H)
cond_zv = np.linalg.cond(H_zv)
print("cond, cond_zv: {}, {}".format(cond_nb, cond_zv))
eig = np.linalg.eigvalsh(H)
eig_zv = np.linalg.eigvalsh(H_zv)


# rcond = 10**(-2)
#theta = np.linalg.solve(H, b)
theta = np.linalg.lstsq(H, b, rcond=10**(-5))[0]
#theta_zv = - np.linalg.solve(H_zv, b_zv)
theta_zv = - np.linalg.lstsq(H_zv, b_zv, rcond=10**(-5))[0]
#
#np.savez('2d_mix_coeffs.npz', 
#         H=H, H_zv=H_zv, b=b, b_zv=b_zv, 
#         theta=theta, theta_zv=theta_zv)

# ---------------

#npzfile = np.load('2d_mix_coeffs.npz')
#H = npzfile['H']
#H_zv = npzfile['H_zv']
#b_zv = npzfile['b_zv']
#b = npzfile['b']
#theta = npzfile['theta']
#theta_zv = npzfile['theta_zv']

#tab_eps = [np.arange(1.0, 0.0, step=-0.1, dtype=float)*10.0**(-i) 
#            for i in np.arange(1, 7, dtype=int)]
#tab_eps = np.concatenate(tab_eps)


for rcond in [10.0**(-i) for i in np.arange(2, 8)]:
  theta = np.linalg.lstsq(H, b, rcond=rcond)[0]
  theta_zv = - np.linalg.lstsq(H_zv, b_zv, rcond=rcond)[0]
  
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
  
#plt.contour(xv, yv, approx_dxpois_mesh)
#plt.colorbar()
#plt.title("approx_dxpois")
#plt.show()
#
#plt.contour(xv, yv, approx_dypois_mesh)
#plt.colorbar()
#plt.title("approx_dypois")
#plt.show()
#
#plt.contour(xv, yv, approx_dxpois_zv)
#plt.title("approx_dxpois_zv")
#plt.colorbar()
#plt.show()
#
#plt.contour(xv, yv, approx_dypois_zv)
#plt.title("approx_dypois_zv")
#plt.colorbar()
#plt.show()
#
#plt.contour(xv, yv, approx_Lpois)
#plt.title("approx_Lpois")
#plt.colorbar()
#plt.show()
#
#plt.contour(xv, yv, approx_Lpois_zv)
#plt.title("approx_Lpois_zv")
#plt.colorbar()
#plt.show()
  
  print('----------------')
  print('rcond :{}'.format(rcond))
  print('----------------')
  
  var_cv = ((approx_dxpois_mesh-dxpois)**2 + (approx_dypois_mesh-dypois)**2)*pi_mesh
  var_cv = var_cv[:-1, :-1]
  area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
  var_cv = 2*np.sum(var_cv)*area
  print('var_cv: {}'.format(var_cv))
  
  var_zv = ((approx_dxpois_zv-dxpois)**2 + (approx_dypois_zv-dypois)**2)*pi_mesh
  var_zv = var_zv[:-1, :-1]
  area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
  var_zv = 2*np.sum(var_zv)*area
  print('var_zv: {}'.format(var_zv))

# Better understanding.

#def varasymp(theta):
#  a_dxpois = np.zeros(dx_psi_mesh[0].shape)
#  a_dypois = np.zeros(dy_psi_mesh[0].shape)
#  for i in np.arange(p**2):
#    a_dxpois += theta[i] * dx_psi_mesh[i]  
#    a_dypois += theta[i] * dy_psi_mesh[i]  
#  a_var = ((a_dxpois-dxpois)**2 + (a_dypois-dypois)**2)*pi_mesh
#  a_var = a_var[:-1, :-1]
#  area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
#  a_var = 2*np.sum(a_var)*area
#  return a_var
#
#def varasymp_gaus(theta):
#  a_dxpois = np.zeros(psi_mesh[0].shape)
#  a_dypois = np.zeros(psi_mesh[0].shape)
#  for i in np.arange(p**2):
#    a_dxpois += theta[i] * psi_mesh[i]  
#    a_dypois += theta[i] * psi_mesh[i]  
#  a_var = ((a_dxpois-dxpois)**2 + (a_dypois-dypois)**2)*pi_mesh
#  a_var = a_var[:-1, :-1]
#  area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
#  a_var = 2*np.sum(a_var)*area
#  return a_var
#
#res = minimize(varasymp, theta)
#res_gaus = minimize(varasymp_gaus, theta)

#%% Try to approximate \tilde{f} in L_2(\pi) using directly \nabla \theta_{zv}^{T} \psi

Lx_psi = np.zeros((meshsize, meshsize, nb_bases**2))
Ly_psi = np.zeros((meshsize, meshsize, nb_bases**2))
dxpsi = np.zeros((meshsize, meshsize, nb_bases**2))
dypsi = np.zeros((meshsize, meshsize, nb_bases**2))
for k, mu in enumerate(mu_vec):
  dxp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  dyp = (2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  dxpsi[:, :, k] = dxp
  dypsi[:, :, k] = dyp
  dxdxpsi = -(2*np.pi*s2)**(-1)*s2**(-1)*(xv-mu[0])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  dydypsi = -(2*np.pi*s2)**(-1)*s2**(-1)*(yv-mu[1])*np.exp(-((xv-mu[0])**2 + (yv-mu[1])**2)/(2*s2))
  
  dxU_dxpsi = dxU_mesh * dxp
  dyU_dypsi = dyU_mesh * dyp
  
  Lx_psi[:, :, k] = - dxU_dxpsi + dxdxpsi
  Ly_psi[:, :, k] = - dyU_dypsi + dydypsi

area = (xv[0, 1] - xv[0, 0])*(yv[1, 0] - yv[0, 0])
f_tilde_mesh = - Lpois

n_mu_vec = len(mu_vec)
npzfile = np.load('Hb_nabla_zv.npz')
H_zv = npzfile['H_zv']
b_zv = npzfile['b_zv']
  
res = np.linalg.lstsq(H_zv, b_zv, rcond=10**(-5))
theta_zv = - res[0]

approx_dxpois_zv = np.dot(dxpsi, theta_zv[:n_mu_vec])
approx_dypois_zv = np.dot(dypsi, theta_zv[n_mu_vec:])

#%% figures

# Weighted by \pi. ----------------

fig = plt.figure(figsize=(16,24))
plt.rcParams.update({'font.size': 13}) # default 10
plt.subplot(3,2,1)
plt.pcolormesh(xv, yv, dxpois*pi_mesh**(1/2))
plt.title(r"$\sqrt{\pi} \partial_1 \hat{f}$")
plt.colorbar()
plt.subplot(3,2,2)
plt.pcolormesh(xv, yv, dypois*pi_mesh**(1/2))
plt.title(r"$\sqrt{\pi} \partial_2 \hat{f}$")
plt.colorbar()
plt.subplot(3,2,3)
plt.pcolormesh(xv, yv, approx_dxpois_mesh*pi_mesh**(1/2))
plt.colorbar()
plt.title(r"$\sqrt{\pi} \partial_1 (\theta^*)^{T} \psi$")
plt.subplot(3,2,4)
plt.pcolormesh(xv, yv, approx_dypois_mesh*pi_mesh**(1/2)) #, vmin=0.0, vmax=0.45)
plt.colorbar()
plt.title(r"$\sqrt{\pi} \partial_2 (\theta^*)^{T} \psi$")
plt.subplot(3,2,5)
plt.pcolormesh(xv, yv, approx_dxpois_zv*pi_mesh**(1/2))
plt.colorbar()
plt.title(r"$\sqrt{\pi} \partial_1 (\theta^*_{zv})^{T} \psi$")
plt.subplot(3,2,6)
plt.pcolormesh(xv, yv, approx_dypois_zv*pi_mesh**(1/2))
plt.colorbar()
plt.title(r"$\sqrt{\pi} \partial_2 (\theta^*_{zv})^{T} \psi$")
plt.show()
fig.savefig("approx_pi_dpois_nabla_2d_rcond5.jpeg", bbox_inches='tight')

fig = plt.figure(figsize=(16,16))
plt.rcParams.update({'font.size': 13}) # default 10
plt.subplot(2,2,1)
plt.pcolormesh(xv, yv, f_tilde_mesh*pi_mesh**(1/2))
plt.title(r"$\sqrt{\pi} \tilde{f}$")
plt.colorbar()
plt.subplot(2,2,2)
plt.pcolormesh(xv, yv, -approx_Lpois*pi_mesh**(1/2))
plt.title(r"$- \sqrt{\pi} \mathcal{L} (\theta^*)^{T} \psi$")
plt.colorbar()
plt.subplot(2,2,3)
plt.pcolormesh(xv, yv, -approx_Lpois_zv*pi_mesh**(1/2))
plt.title(r"$- \sqrt{\pi} \mathcal{L} (\theta^*_{zv})^{T} \psi$")
plt.colorbar()
plt.show()
fig.savefig("approx_pi_Lpois_2d_nabla_rcond5.jpeg", bbox_inches='tight')

# Unweighted. ------------------

fig = plt.figure(figsize=(16,24))
plt.rcParams.update({'font.size': 13}) # default 10
plt.subplot(3,2,1)
plt.pcolormesh(xv, yv, dxpois)
plt.title(r"$\partial_1 \hat{f}$")
plt.colorbar()
plt.subplot(3,2,2)
plt.pcolormesh(xv, yv, dypois)
plt.title(r"$\partial_2 \hat{f}$")
plt.colorbar()
plt.subplot(3,2,3)
plt.pcolormesh(xv, yv, approx_dxpois_mesh)
plt.colorbar()
plt.title(r"$\partial_1 (\theta^*)^{T} \psi$")
plt.subplot(3,2,4)
plt.pcolormesh(xv, yv, approx_dypois_mesh)
plt.colorbar()
plt.title(r"$\partial_2 (\theta^*)^{T} \psi$")
plt.subplot(3,2,5)
plt.pcolormesh(xv, yv, approx_dxpois_zv)
plt.colorbar()
plt.title(r"$\partial_1 (\theta^*_{zv})^{T} \psi$")
plt.subplot(3,2,6)
plt.pcolormesh(xv, yv, approx_dypois_zv)
plt.colorbar()
plt.title(r"$\partial_2 (\theta^*_{zv})^{T} \psi$")
plt.show()
fig.savefig("approx_dpois_2d_nabla_rcond5.jpeg", bbox_inches='tight')


fig = plt.figure(figsize=(16,16))
plt.rcParams.update({'font.size': 13}) # default 10
plt.subplot(2,2,1)
plt.pcolormesh(xv, yv, f_tilde_mesh)
plt.title(r"$\tilde{f}$")
plt.colorbar()
plt.subplot(2,2,2)
plt.pcolormesh(xv, yv, -approx_Lpois)
plt.title(r"$- \mathcal{L} (\theta^*)^{T} \psi$")
plt.colorbar()
plt.subplot(2,2,3)
plt.pcolormesh(xv, yv, -approx_Lpois_zv)
plt.title(r"$- \mathcal{L} (\theta^*_{zv})^{T} \psi$")
plt.colorbar()
plt.show()
fig.savefig("approx_Lpois_2d_nabla_rcond5.jpeg", bbox_inches='tight')

#%% Mask for the support of \pi

pi_support = pi_mesh >= 10**(-2)
#pi_masked = ma.masked_less_equal(pi_mesh, 10**(-2))
#pi_support = pi_masked.filled(0)

diff_dxpois_cv = np.max(np.absolute((approx_dxpois_mesh - dxpois)*pi_support))
diff_dypois_cv = np.max(np.absolute((approx_dypois_mesh - dxpois)*pi_support))
diff_dxpois_zv = np.max(np.absolute((approx_dxpois_zv - dxpois)*pi_support))
diff_dypois_zv = np.max(np.absolute((approx_dypois_zv - dxpois)*pi_support))

diff_Lpois_cv = np.max(np.absolute((f_tilde_mesh + approx_Lpois)*pi_support))
diff_Lpois_zv = np.max(np.absolute((f_tilde_mesh + approx_Lpois_zv)*pi_support))


fig = plt.figure(figsize=(16,24))
plt.rcParams.update({'font.size': 13}) # default 10
plt.subplot(3,2,1)
plt.contour(xv, yv, dxpois)
plt.title(r"$\partial_1 \hat{f}$")
plt.colorbar()
plt.subplot(3,2,2)
plt.contour(xv, yv, dypois)
plt.title(r"$\partial_2 \hat{f}$")
plt.colorbar()
plt.subplot(3,2,3)
plt.contour(xv, yv, (approx_dxpois_mesh - dxpois)*pi_support)
plt.colorbar()
plt.title(r"$\partial_1 (\theta^*)^{T} \psi$")
plt.subplot(3,2,4)
plt.contour(xv, yv, (approx_dypois_mesh - dxpois)*pi_support)
plt.colorbar()
plt.title(r"$\partial_2 (\theta^*)^{T} \psi$")
plt.subplot(3,2,5)
plt.contour(xv, yv, (approx_dxpois_zv - dxpois)*pi_support)
plt.colorbar()
plt.title(r"$\partial_1 (\theta^*_{zv})^{T} \psi$")
plt.subplot(3,2,6)
plt.contour(xv, yv, (approx_dypois_zv - dxpois)*pi_support)
plt.colorbar()
plt.title(r"$\partial_2 (\theta^*_{zv})^{T} \psi$")
plt.show()


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

#def MALA(step, N, n, U, gradU):
#  traj = np.zeros(n)
#  traj_grad = np.zeros(n)
#  x = 0. # np.random.normal(scale=5.0, size=d)
##    accept = 0
#  for k in np.arange(N):
#    y = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal()
#    logratio = -U(y)+U(x) + (1./(4*step))*((y-x+step*gradU(x))**2 \
#                                         - (x-y+step*gradU(y))**2)
#    if np.log(np.random.uniform())<=logratio:
#      x = y
#  for k in np.arange(n):
#    traj[k] = x
#    traj_grad[k] = gradU(x)
#    y = x - step * gradU(x) + np.sqrt(2*step)*np.random.normal()
#    logratio = -U(y)+U(x)+(1./(4*step))*((y-x+step*gradU(x))**2 \
#                                         -(x-y+step*gradU(y))**2)
#    if np.log(np.random.uniform())<=logratio:
#      x = y
##            accept +=1
##    print(np.float(accept) / n)
#  return (traj, traj_grad)
#
#def RWM(step, N, n, U, gradU):
#  traj = np.zeros(n)
#  traj_grad = np.zeros(n)
#  x = 0. # np.random.normal(scale=5.0, size=d)
#  for k in np.arange(N):
#    y = x + np.sqrt(2*step)*np.random.normal()
#    logratio = -U(y)+U(x)
#    if np.log(np.random.uniform())<=logratio:
#      x = y
#  for k in np.arange(n):
#    traj[k]=x
#    traj_grad[k]=gradU(x)
#    y = x + np.sqrt(2*step)*np.random.normal()
#    logratio = -U(y)+U(x)
#    if np.log(np.random.uniform())<=logratio:
#      x = y
#  return (traj, traj_grad)

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
#n_simu = 10

#tab_avar = np.zeros(n_simu)
#tab_avar_cv = np.zeros(n_simu)
#tab_avar_zv = np.zeros(n_simu)
#
#for k in np.arange(n_simu):

traj, traj_grad = ULA(step,N,n,dU)

ns = analyse_samples(traj,traj_grad)
ns_cv = analyse_samples(traj,traj_grad,theta=theta)
ns_zv = analyse_samples(traj,traj_grad,theta=theta_zv)
ns_cv = analyse_samples(traj,traj_grad,theta=theta,gaus=False)
ns_zv = analyse_samples(traj,traj_grad,theta=theta_zv,gaus=False)

#avar, avar_cv, avar_zv = ns[0][1], ns_cv[0][1], ns_zv[0][1]

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
#fig.savefig("autocorrelations_2d.pdf", bbox_inches='tight')



