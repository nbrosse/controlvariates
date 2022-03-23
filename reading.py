# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as spstats
from scipy import signal
import sklearn.preprocessing as skpre
#import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pickle
#from multiprocessing import Pool


#%% Reading logit

def tab(log_ula):
    nc = len(log_ula)
    p, d = log_ula[0].shape
    tab = np.empty((p,nc,d))
    for i in np.arange(nc):
        tab[:,i,:] = log_ula[i]
    return tab

with open('archives//log_ula_nc100_N5_n6.pkl', 'rb') as f:
    log_ula = pickle.load(f, encoding='latin1')

tab_log_ula = tab(log_ula)

with open('archives//log_mala_nc100_N5_n6.pkl', 'rb') as f:
    log_mala = pickle.load(f, encoding='latin1')

tab_log_mala = tab(log_mala)

with open('archives//log_rwm_nc100_N5_n6.pkl', 'rb') as f:
    log_rwm = pickle.load(f, encoding='latin1')

tab_log_rwm = tab(log_rwm)

p, nc, d = tab_log_ula.shape
tab_log = np.stack([tab_log_ula, tab_log_mala, tab_log_rwm], axis=-1)

tab_labels = ["O", "CV-1", "CV-2", "ZV-1", "ZV-2"]
tab_titles = ["ULA", "MALA", "RWM"]

ldfp = []
indices = np.array([4,5,6,7]) #indices = np.array([0,1,4,5])
#xnames = ["x_1", "x_2", "x_1^2", "x_2^2"]
xnames = ["x_1^2", "x_2^2", "x_3^2", "x_4^2"]
#xnames_tex = ["$x_1$", "$x_2$", "$x_1^2$", "$x_2^2$"]

for l in np.arange(4):
    a = tab_log[::2,:,indices[l],:]
    a = np.swapaxes(a, 0, 1)
    df = np.empty((a.size,3), dtype=object)
    df1 = np.zeros(a.size)
    df2 = np.empty(a.size, dtype=np.dtype('U25'))
    df3 = np.empty(a.size, dtype=np.dtype('U25'))
    compt = 0
    nb_methods = 5
    tab_labels = ["O", "CV-1", "CV-2", "ZV-1", "ZV-2"]
    for i in np.arange(nc):
        for j in np.arange(nb_methods):
            for k in np.arange(3):
                df1[compt] = a[i,j,k]
                df2[compt] = tab_labels[j] 
                df3[compt] = tab_titles[k]
                compt +=1

    df = np.rec.fromarrays((df1,df2,df3), names=(xnames[l], "method", "algorithm"))    
    dfp = pd.DataFrame(df)    
    ldfp.append(dfp)

#for l in np.arange(4):
#    if l==0:
#        a = tab_log[::2,:,0,:]
#    elif l==1:
#        a = tab_log[::2,:,0,:]
#        a = a[[-3,-1],:,:]
#    elif l==2:
#        a = tab_log[::2,:,4,:]
#    else:
#        a = tab_log[::2,:,4,:]
#        a = a[[-3,-1],:,:]
##    a = tab_log[::2,:,indices[l],:]
#    a = np.swapaxes(a, 0, 1)
#    df = np.empty((a.size,3), dtype=object)
#    df1 = np.zeros(a.size)
#    df2 = np.empty(a.size, dtype=np.dtype('U25'))
#    df3 = np.empty(a.size, dtype=np.dtype('U25'))
#    compt = 0
#    if l%2==0:
#        nb_methods = 5
#        tab_labels = ["O", "CV-1", "CV-2", "ZV-1", "ZV-2"]
#    else:
#        nb_methods = 2
#        tab_labels = ["CV-2", "ZV-2"]
#    for i in np.arange(nc):
#        for j in np.arange(nb_methods):
#            for k in np.arange(3):
#                df1[compt] = a[i,j,k]
#                df2[compt] = tab_labels[j] 
#                df3[compt] = tab_titles[k]
#                compt +=1
#    
#    df = np.rec.fromarrays((df1,df2,df3), names=(xnames[l], "method", "algorithm"))    
#    dfp = pd.DataFrame(df)    
#    ldfp.append(dfp)

f = plt.figure(figsize=(16,16))
for k in np.arange(4):
    plt.subplot(2,2,k+1)
    ax = sns.violinplot(x="method", y=xnames[k], hue="algorithm", data=ldfp[k] ,palette="muted")
    #ax.set_title("Titre")
    ax.set_ylabel(r"$" + xnames[k] + "$")
#    ax.set_xlabel()

plt.show()
f.savefig("log-sb-1-2.pdf", bbox_inches='tight')
# \beta_1, ..., \beta_d

f = plt.figure(figsize=(16,16))

for k in np.arange(12):
#    plt.subplot(int('43'+str(k+1)))
    plt.subplot(4,3,k+1)
    temp = np.transpose(tab_log[::2,:,k // 3,k % 3])
#    if (k % 3)==0:
#        mini, maxi = np.min(temp), np.max(temp)
    plt.boxplot(temp, labels=tab_labels)
#    plt.ylim(mini, maxi)
    if (k % 3)==0:
        st = f"$x_{str(k // 3 +1)}$"
        plt.ylabel(st)
    plt.title(tab_titles[k % 3])
    plt.grid(True)

#plt.suptitle("Ill conditionned Gaussian, first coordinate, error " \
#          + indi +" moment, N=10**6, dimension " + str(d) + " , x0=" + str(x0Tab[x0_ind]))
plt.show()
f.savefig("log-1-1.pdf", bbox_inches='tight')

# \beta^2_1, ..., \beta^2_d

f = plt.figure(figsize=(16,16))

for k in np.arange(12):
#    plt.subplot(int('43'+str(k+1)))
    plt.subplot(4,3,k+1)
    temp = np.transpose(tab_log[::2,:,4 + k // 3,k % 3])
    plt.boxplot(temp, labels=tab_labels)
    if (k % 3)==0:
        st = f"$(x_{str(k // 3 +1)})^2$"
        plt.ylabel(st)
    plt.title(tab_titles[k % 3])
    plt.grid(True)

#plt.suptitle("Ill conditionned Gaussian, first coordinate, error " \
#          + indi +" moment, N=10**6, dimension " + str(d) + " , x0=" + str(x0Tab[x0_ind]))
plt.show()
f.savefig("log-2-1.pdf", bbox_inches='tight')


#%% Reading probit

with open('archives//pro_ula_nc100_N5_n6.pkl', 'rb') as f:
    pro_ula = pickle.load(f, encoding='latin1')

tab_pro_ula = tab(pro_ula)

with open('archives//pro_mala_nc100_N5_n6.pkl', 'rb') as f:
    pro_mala = pickle.load(f, encoding='latin1')

tab_pro_mala = tab(pro_mala)

with open('archives//pro_rwm_nc100_N5_n6.pkl', 'rb') as f:
    pro_rwm = pickle.load(f, encoding='latin1')

tab_pro_rwm = tab(pro_rwm)

p, nc, d = tab_pro_ula.shape

tab_pro = np.stack([tab_pro_ula, tab_pro_mala, tab_pro_rwm], axis=-1)

#### Array of figures
#
#mean_tab_pro = np.mean(tab_pro, axis=1)
#mean_tab_pro = mean_tab_pro[1::2,:,:]
#VRF = np.divide(mean_tab_pro[0,:,:], mean_tab_pro[1:,:,:])
#tabl_pro = np.insert(mean_tab_pro, [1,2,3,4], VRF, axis=0)
##tabl_log = np.swapaxes(tabl_log,0,2)
#
#L=[]
#for i in np.arange(d):
#    L.append(np.transpose(tabl_pro[:,i,:]))
#    
#l = np.vstack(L)
#
##np.savetxt("mydata.csv", l, fmt='%10.5f', delimiter=' & ', newline=' \\\\\n')
#np.savetxt("pro_tab.csv", l, fmt='%10.2g', delimiter=' & ', newline=' \\\\\n')


##### END


tab_labels = ["O", "CV-1", "CV-2", "ZV-1", "ZV-2"]
tab_titles = ["ULA", "MALA", "RWM"]

ldfp = []
indices = np.array([3,4,5]) #indices = np.array([0,1,4,5])
#xnames = ["x_1", "x_2", "x_1^2", "x_2^2"]
xnames = ["x_1^2", "x_2^2", "x_3^2"]
nb_methods = 5
#xnames_tex = ["$x_1$", "$x_2$", "$x_1^2$", "$x_2^2$"]

for l in np.arange(3):
    a = tab_pro[::2,:,indices[l],:]
    a = np.swapaxes(a, 0, 1)
    df = np.empty((a.size,3), dtype=object)
    df1 = np.zeros(a.size)
    df2 = np.empty(a.size, dtype=np.dtype('U25'))
    df3 = np.empty(a.size, dtype=np.dtype('U25'))
    compt = 0
    tab_labels = ["O", "CV-1", "CV-2", "ZV-1", "ZV-2"]
    for i in np.arange(nc):
        for j in np.arange(nb_methods):
            for k in np.arange(3):
                df1[compt] = a[i,j,k]
                df2[compt] = tab_labels[j] 
                df3[compt] = tab_titles[k]
                compt +=1

    df = np.rec.fromarrays((df1,df2,df3), names=(xnames[l], "method", "algorithm"))
    dfp = pd.DataFrame(df)
    ldfp.append(dfp)

f = plt.figure(figsize=(16,16))
for k in np.arange(3):
    plt.subplot(2,2,k+1)
    ax = sns.violinplot(x="method", y=xnames[k], hue="algorithm", data=ldfp[k] ,palette="muted")
    #ax.set_title("Titre")
    ax.set_ylabel(r"$" + xnames[k] + "$")
#    ax.set_xlabel()

plt.show()
f.savefig("pro-sb-1-2.pdf", bbox_inches='tight')

######### OLD

tab_labels = ["O", "CV-1", "CV-2", "ZV-1", "ZV-2"]
tab_titles = ["ULA", "MALA", "RWM"]

# \beta_1, ..., \beta_d

f = plt.figure(figsize=(16,12))

for k in np.arange(9):
#    plt.subplot(int('43'+str(k+1)))
    plt.subplot(3,3,k+1)
    temp = np.transpose(tab_pro[::2,:,k // 3,k % 3])
    plt.boxplot(temp, labels=tab_labels)
    if (k % 3)==0:
        st = f"$x_{str(k // 3 +1)}$"
        plt.ylabel(st)
    plt.title(tab_titles[k % 3])
    plt.grid(True)

#plt.suptitle("Ill conditionned Gaussian, first coordinate, error " \
#          + indi +" moment, N=10**6, dimension " + str(d) + " , x0=" + str(x0Tab[x0_ind]))
plt.show()
f.savefig("pro-1-1.pdf", bbox_inches='tight')

# \beta^2_1, ..., \beta^2_d

f = plt.figure(figsize=(16,12))

for k in np.arange(9):
#    plt.subplot(int('43'+str(k+1)))
    plt.subplot(3,3,k+1)
    temp = np.transpose(tab_pro[::2,:,3 + k // 3,k % 3])
    plt.boxplot(temp, labels=tab_labels)
    if (k % 3)==0:
        st = f"$(x_{str(k // 3 +1)})^2$"
        plt.ylabel(st)
    plt.title(tab_titles[k % 3])
    plt.grid(True)

#plt.suptitle("Ill conditionned Gaussian, first coordinate, error " \
#          + indi +" moment, N=10**6, dimension " + str(d) + " , x0=" + str(x0Tab[x0_ind]))
plt.show()
f.savefig("pro-2-1.pdf", bbox_inches='tight')