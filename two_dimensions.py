# -*- coding: utf-8 -*-

#%% Packages

import numpy as np
#import scipy.stats as spstats
from scipy import signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#import seaborn as sns
import itertools


#%% mixture of 2 gaussians
  
mu_1 = np.array([-1., 0.])
mu_2 = np.array([1., 0.])
sigma2 = 0.5
nb_bases = 5
bound = 3

for bound in np.arange(1, 5, step=0.2):
    
  print('--------------------------')
  print('bound: {}'.format(bound))
  print('--------------------------')
  
  meshsize = 50
  x = np.linspace(-bound, bound, meshsize)
  y = np.linspace(-bound, bound, meshsize)
  xv, yv = np.meshgrid(x, y)
  
  ind = 0
  dx_pi_dxpois = np.zeros((meshsize, meshsize, nb_bases**2))
  dy_pi_dypois = np.zeros((meshsize, meshsize, nb_bases**2))
  pi_dpois = np.zeros((meshsize, meshsize, nb_bases**2))
  for kx in np.arange(1, nb_bases+1):
    for ky in np.arange(1, nb_bases+1):
      dx = (kx*np.pi/2./bound)*np.cos(kx*np.pi*(xv+bound)/(2*bound))*np.sin(ky*np.pi*(yv+bound)/(2*bound))
      dy = (ky*np.pi/2./bound)*np.sin(kx*np.pi*(xv+bound)/(2*bound))*np.cos(ky*np.pi*(yv+bound)/(2*bound))
      dx_pi_dxpois[:, :, ind] = dx
      dy_pi_dypois[:, :, ind] = dy
      pi_dpois[:, :, ind] = np.sin(kx*np.pi*(xv+bound)/(2*bound))*np.sin(ky*np.pi*(yv+bound)/(2*bound))
      ind += 1
      
  dx_pi_dxpois_rect = np.reshape(dx_pi_dxpois, (meshsize**2, nb_bases**2))    
  dy_pi_dypois_rect = np.reshape(dy_pi_dypois, (meshsize**2, nb_bases**2))    
  
  div_pi_pois = np.hstack((dx_pi_dxpois_rect, dy_pi_dypois_rect))
  
  def f(y, x):
    return x + y**3 + np.sin(x) + np.cos(y)
          
  def pi_yx(y, x):
    return pi(np.array([x, y]))
    
  def pi(x):
    return 0.5*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2)) / (2*np.pi*sigma2) \
            + 0.5*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2)) / (2*np.pi*sigma2)
  
  
  pi_mesh = 0.5*np.exp(-((xv-mu_1[0])**2+(yv-mu_1[1])**2)/(2*sigma2)) / (2*np.pi*sigma2) \
            + 0.5*np.exp(-((xv-mu_2[0])**2+(yv-mu_2[1])**2)/(2*sigma2)) / (2*np.pi*sigma2)  
  
  pi_f = integrate.dblquad(lambda y, x: f(y, x)*pi_yx(y, x), -bound, bound, lambda x: -bound, lambda x :bound)[0]
  f_tilde_mesh = xv + yv**3 + np.sin(xv) + np.cos(yv) - pi_f
  f_tilde_pi_mesh = (f_tilde_mesh * pi_mesh).flatten()
  
  coeffs_pois = np.linalg.lstsq(div_pi_pois, - f_tilde_pi_mesh, rcond=None)[0]
  
  pi_dxpois = np.dot(pi_dpois, coeffs_pois[:nb_bases**2])
  pi_dypois = np.dot(pi_dpois, coeffs_pois[nb_bases**2:])
  
  dxpois = np.divide(pi_dxpois, pi_mesh)
  dypois = np.divide(pi_dypois, pi_mesh)
  
  #fig = plt.figure()
  #ax = fig.gca(projection='3d')
  #ax.plot_surface(xv, yv, f_tilde_mesh) # cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
  #ax.plot_surface(xv[10:40, 10:40], yv[10:40, 10:40], dxpois[10:40, 10:40]) # cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
  
  #plt.contour(xv, yv, f_tilde_mesh)
  #plt.colorbar()
  #plt.show()  
  
  plt.contour(xv, yv, pi_dypois)
  plt.colorbar()
  plt.show()
  
  plt.contour(xv, yv, pi_dxpois)
  plt.colorbar()
  plt.show()
  
             
  def U(x):
    return -np.log(pi(x))
  
  def dU(x):
    dpi = (x-mu_1)*np.exp(-np.linalg.norm(x-mu_1)**2/(2*sigma2))
    dpi += (x-mu_2)*np.exp(-np.linalg.norm(x-mu_2)**2/(2*sigma2))
    dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
    return - dpi / pi(x)
  
  var = dxpois*pi_dxpois + dypois*pi_dypois
  var = var[:-1, :-1]
  area = (xv[0, 1] - xv[0, 0])**2
  var = np.sum(var)*area
  print('var: {}'.format(var))

#%% Basis of functions

p = 5
s2 = 1.0
mu_vec = [np.array([mu1, mu2]) for mu1, mu2 in itertools.product(
    np.linspace(-bound, bound, num=p), np.linspace(-bound, bound, num=p))]


psi = [(lambda mu1, mu2: lambda y, x: (2*np.pi*s2)**(-1)*np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2)))(mu[0], mu[1]) for mu in mu_vec]

psi_mesh = [(2*np.pi*s2)**(-1)*np.exp(-((xv-mu[0])**2+(yv-mu[1])**2)/(2*s2)) for mu in mu_vec]

dx_psi = [(lambda mu1, mu2: lambda y, x: -(2*np.pi*s2)**(-1)*s2**(-1)*(x-mu1)*np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2)))(mu[0], mu[1]) for mu in mu_vec]
dy_psi = [(lambda mu1, mu2: lambda y, x: -(2*np.pi*s2)**(-1)*s2**(-1)*(y-mu2)*np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2)))(mu[0], mu[1]) for mu in mu_vec]

Lpsi = [(lambda i: (lambda mu1, mu2: lambda y, x: - dx_psi[i](y, x)*dU([x, y])[0] - dy_psi[i](y, x)*dU([x, y])[1] + (2*np.pi*s2)**(-1)*s2**(-1)*
         np.exp(-((x-mu1)**2+(y-mu2)**2)/(2*s2))*(((x-mu1)**2+(y-mu2)**2)/s2 - 2.))(mu[0], mu[1]))(i) for (i, mu) in zip(np.arange(p**2), mu_vec)]

#for i in np.arange(p**2):
#  plt.contour(xv, yv, psi_mesh[i])
#  plt.show()

H = np.zeros((p**2, p**2))
for i in np.arange(p**2):
  for j in np.arange(p**2):
    H[i,j] = integrate.dblquad(lambda y, x: (dx_psi[i](y, x)*dx_psi[j](y, x)+
     dy_psi[i](y, x)*dy_psi[j](y, x))*pi_yx(y, x), -bound, bound, lambda x:-bound, lambda x:bound)[0]

H_zv = np.zeros((p**2, p**2))
for i in np.arange(p**2):
  for j in np.arange(p**2):
    H_zv[i,j] = integrate.dblquad(lambda y, x: Lpsi[i](y, x)*Lpsi[j](y, x)*pi_yx(y, x), 
        -bound, bound, lambda x:-bound, lambda x:bound)[0]

b = np.zeros(p**2)
for i in np.arange(p**2):
  b[i] = integrate.dblquad(lambda y, x: (f(y, x) - pi_f)*psi[i](y, x)*pi_yx(y, x), 
   -bound, bound, lambda x:-bound, lambda x:bound)[0]

b_zv = np.zeros(p**2)
for i in np.arange(p**2):
  b_zv[i] = integrate.dblquad(lambda y, x: (f(y, x) - pi_f)*Lpsi[i](y, x)*pi_yx(y, x), 
      -bound, bound, lambda x:-bound, lambda x:bound)[0]

cond_nb = np.linalg.cond(H)
eig = np.linalg.eigvalsh(H)
eig_zv = np.linalg.eigvalsh(H_zv)

theta = np.linalg.solve(H, b)
theta_zv = - np.linalg.solve(H_zv, b_zv)

np.savez('Hbtheta.npz', H=H, H_zv=H_zv, b=b, b_zv=b_zv, 
         theta=theta, theta_zv=theta_zv)

npzfile = np.load('Hbtheta.npz')
H = npzfile['H']
H_zv = npzfile['H_zv']
b_zv = npzfile['b_zv']
b = npzfile['b']
theta = npzfile['theta']
theta_zv = npzfile['theta_zv']


#def approx_dpois(y, x):
#  dx, dy = 0., 0.
#  for i in np.arange(p**2):
#    dx += theta[i] * dx_psi[i](t)
#    dy += theta[i] * dy_psi[i](t)
#  return res
#
#def approx_Lpois(t):
#  res = 0.
#  for i in np.arange(p):
#    res += theta[i] * Lpsi[i](t)
#  return res
#
#def approx_dpois_L(t):
#  res = 0.
#  for i in np.arange(p):
#    res += theta_L[i] * dpsi[i](t)
#  return res
#
#def approx_Lpois_L(t):
#  res = 0.
#  for i in np.arange(p):
#    res += theta_L[i] * Lpsi[i](t)
#  return res
#
#var_cv = 2*integrate.quad(lambda x: (dpois(x) - approx_dpois(x))**2*pi(x), -bound, bound)[0]
#var_zv = 2*integrate.quad(lambda x: (dpois(x) - approx_dpois_L(x))**2*pi(x), -bound, bound)[0]

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

def normalSamples(traj, traj_grad, theta=None):
  n = traj.shape[0]
  if theta is None:
    samples = traj[:, 0] + np.power(traj[:, 1], 3) + np.sin(traj[:, 0]) + np.cos(traj[:,1])
  else:
    x = traj[:, 0]
    y = traj[:, 1]
    laplacien_psi = [(2*np.pi*s2)**(-1)*s2**(-1)*np.exp(-((x-mu[0])**2+(y-mu[1])**2)/(2*s2))*(((x-mu[0])**2+(y-mu[1])**2)/s2 - 2.) for mu in mu_vec]
    laplacien_psi = np.vstack(tuple(laplacien_psi)).T
    dx_psi= [-(2*np.pi*s2)**(-1)*s2**(-1)*(x-mu[0])*np.exp(-((x-mu[0])**2+(y-mu[1])**2)/(2*s2)) for mu in mu_vec]
    dy_psi= [-(2*np.pi*s2)**(-1)*s2**(-1)*(y-mu[1])*np.exp(-((x-mu[0])**2+(y-mu[1])**2)/(2*s2)) for mu in mu_vec]
    dx_psi = np.vstack(tuple(dx_psi)).T
    dy_psi = np.vstack(tuple(dy_psi)).T
    
    laplacien_psi = np.dot(laplacien_psi, theta)
    dx_psi = np.dot(dx_psi, theta)
    dy_psi = np.dot(dy_psi, theta)

    dpi = (traj-mu_1)*(np.repeat(np.exp(-np.sum((traj-mu_1)**2, axis=1)/(2*sigma2)), 2).reshape(n, 2))
    dpi += (traj-mu_2)*(np.repeat(np.exp(-np.sum((traj-mu_2)**2, axis=1)/(2*sigma2)), 2).reshape(n, 2))
    dpi *= (-0.5)*sigma2**(-1)*(2*np.pi*sigma2)**(-1.)
    
    pi_traj = 0.5*np.exp(-np.sum((traj-mu_1)**2, axis=1)/(2*sigma2)) / (2*np.pi*sigma2) \
          + 0.5*np.exp(-np.sum((traj-mu_2)**2, axis=1)/(2*sigma2)) / (2*np.pi*sigma2)

    dU = - np.divide(dpi, np.repeat(pi_traj, 2).reshape(n, 2))
    
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



ns = normalSamples(traj,traj_grad)
ns_cv = normalSamples(traj,traj_grad,theta=theta)
ns_zv = normalSamples(traj,traj_grad,theta=theta_zv)

a, a_cv, a_zv = ns[1], ns_cv[1], ns_zv[1]

nb_cor = 100

plt.bar(np.arange(nb_cor), a[:nb_cor], label='without')
#plt.title('corr without vr')
#plt.show()
#plt.title('corr with cv')
#plt.show()
plt.bar(np.arange(nb_cor), a_cv[:nb_cor], label='cv')
plt.bar(np.arange(nb_cor), a_zv[:nb_cor], label='zv')
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


