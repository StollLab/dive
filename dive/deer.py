import numpy as np
import math as m

from .constants import *

# This function can be removed?
def dd_gauss(r,r0,fwhm,a=1):
    """
    Calculates a multi-Gauss distance distribution over distance vector r.
    """

    nGauss = np.size(r0)

    if np.size(fwhm)!=nGauss:
        raise ValueError("r0 and fwhm need to have the same number of elements.")
    if np.size(a)!=nGauss:
        raise ValueError("r0 and a need to have the same number of elements.")
    
    sig = np.array(fwhm)/2/m.sqrt(2*m.log(2))

    if nGauss==1:
        P = a*gauss(r,r0,sig)
    else:
        P = np.zeros_like(r)
        for k in range(nGauss):
            P += a[k]*gauss(r,r0[k],sig[k])

    return P


def gauss(r,r0,sig):
    """
    Calculates a single-Gauss distance distribution over distance vector r.
    """
    return m.sqrt(1/2/m.pi)/sig*np.exp(-((r-r0)/sig)**2/2)

def bg_exp(t,k):
    """
    Exponential background decay
    """
    return np.exp(-np.abs(t)*k)

def bg_exp_time(t,tauB):
    """
    Exponential background decay
    """
    return np.exp(-np.abs(t)/tauB)

def bg_hom3d(t,conc,lamb):
    """
    Calculates a background decay due to a homogeneous 3D distribution of spins
    conc: concentration in micromolar
    lamb: modulation depth
    """
    
    conc = conc*1e-6*1e3*NA # umol/L -> mol/L -> mol/m^3 -> spins/m^3
    
    B = np.exp(-8*m.pi**2/9/m.sqrt(3)*lamb*conc*D*abs(t*1e-6))
    return B
