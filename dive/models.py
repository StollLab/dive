import numpy as np
import math as m

# Fundamental constants (CODATA 2018)
NA = 6.02214076e23 # Avogadro constant, mol^-1
muB = 9.2740100783e-24 # Bohr magneton, J/T
mu0 = 1.25663706212e-6 # magnetic constant, N A^-2 = T^2 m^3 J^-1
h = 6.62607015e-34 # Planck constant, J/Hz
ge = 2.00231930436256 # free-electron g factor
hbar = h/2/m.pi # reduced Planck constant, J/(rad/s)
D = (mu0/4/m.pi)*(muB*ge)**2/hbar # dipolar constant, m^3 rad s^-1


def dd_gauss(r,r0,fwhm,A=1):
    """
    Calculates a multi-Gauss distance distribution over distance vector r.
    """
    nGauss = np.size(r0)
    if np.size(fwhm)!=nGauss:
        raise ValueError("r0 and fwhm need to have the same number of elements.")
    if np.size(A)!=nGauss:
        raise ValueError("r0 and A need to have the same number of elements.")
    sig = np.array(fwhm)/2/m.sqrt(2*m.log(2))
    if nGauss==1:
        P = A*gauss(r,r0,sig)
    else:
        P = np.zeros_like(r)
        for k in range(nGauss):
            P += A[k]*gauss(r,r0[k],sig[k])
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
    return np.exp(-abs(t)*k)


def bg_hom3d(t,conc,lamb):
    """
    Calculates a background decay due to a homogeneous 3D distribution of spins
    conc: concentration in micromolar
    lamb: modulation depth
    """
    
    conc = conc*1e-6*1e3*NA # umol/L -> mol/L -> mol/m^3 -> spins/m^3
    
    B = np.exp(-8*m.pi**2/9/m.sqrt(3)*lamb*conc*D*abs(t*1e-6))
    return B
