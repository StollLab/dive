import numpy as np
import math as m
from scipy.special import fresnel

from .constants import *

def addnoise(V,sig):
    """
    Add Gaussian noise with standard deviation sig to signal
    """
    noise = np.random.normal(0,sig,np.size(V))
    Vnoisy = V + noise
    return Vnoisy

def dipolarkernel(t,r):
    """
    K = dipolarkernel(t,r)
    Calculate dipolar kernel matrix.
    Assumes t in microseconds and r in nanometers
    """
    wr = w0/(r*1e-9)**3  # rad s^-1
        
    # Calculation using Fresnel integrals
    nr = np.size(r)
    nt = np.size(t)
    K = np.zeros((nt,nr))
    for ir in range(nr):
        ph = wr[ir]*(np.abs(t)*1e-6)
        kappa = np.sqrt(6*ph/m.pi)
        S, C = fresnel(kappa)
        K[:,ir] = (np.cos(ph)*C+np.sin(ph)*S)/kappa
    
    K[t==0,:] = 1   # fix div by zero
    
    # Include delta-r factor for integration
    if len(r)>1:
        dr = np.mean(np.diff(r))
        K = K*dr
    
    return K
