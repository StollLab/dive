import numpy as np
import math as m

from .constants import *

from typing import Union
from numpy.typing import ArrayLike

def dd_gauss(
    r: ArrayLike, r0: Union[float,ArrayLike], fwhm: Union[float,ArrayLike], 
    a: Union[float,ArrayLike] = 1) -> np.ndarray:
    """Calculates a multi-Gauss distance distribution.
    
    Requires the distance vector r in nanometers and the mean, width,
    and amplitude of each gaussian distribution. The distance 
    distribution will be normalized such that integrates to 1 over r.

    Parameters
    ----------
    r : ArrayLike
        The distance axis to generate P on.
    r0 : float or ArrayLike of float
        The mean(s) of the Gaussian(s).
    fwhm : float or ArrayLike of float
        The full width(s) at half maximum of the Gaussian(s).
    a : float or ArrayLike of float, default=1
        The amplitudes of the Gaussians, for a multi-Gauss distribution.
    
    Returns
    -------
    P : np.ndarray
        A normalized vector of the distance distribution.
    """

    # number of gaussians
    n_gauss = np.size(r0)
    if np.size(fwhm)!= n_gauss:
        raise ValueError("r0 and fwhm need to have the same number of elements.")
    if np.size(a)!= n_gauss:
        raise ValueError("r0 and a need to have the same number of elements.")
    
    # calculate P
    sig = np.array(fwhm)/2/m.sqrt(2*m.log(2))
    if n_gauss==1:
        P = a*gauss(r,r0,sig)
    else:
        P = np.zeros_like(r)
        for k in range(n_gauss):
            P += a[k]*gauss(r,r0[k],sig[k])

    # normalize P
    scale = P.sum() * (r[1]-r[0])
    P /= scale

    return P


def gauss(r: ArrayLike, r0: float, sig: float) -> np.ndarray:
    """Calculates a single-Gauss distance distribution.
    
    Requires the distance vector r in nanometers and the mean and 
    standard deviation of the Gaussian. The distance distribution will
    be normalized such that it integrates to 1 over r.

    Parameters
    ----------
    r : ArrayLike
        The distance axis to generate P on.
    r0 : float
        The mean of the Gaussian.
    sig : float
        The standard deviation of the Gaussian.
    
    Returns
    -------
    P : np.ndarray
        A normalized vector of the distance distribution.
    """
    P = m.sqrt(1/2/m.pi)/sig*np.exp(-((r-r0)/sig)**2/2)
    # normalize P
    scale = P.sum() * (r[1]-r[0])
    P /= scale
    return P

def bg_exp(t: ArrayLike, k: float) -> np.ndarray:
    """Generates an exponential background decay vector from k.

    Parameters
    ----------
    t : ArrayLike
        The time axis to generate the background decay on.
    k : float
        The exponent of the exponential decay.
    
    Returns
    -------
    B : np.ndarray
        The background decay vector.
    """
    B = np.exp(-np.abs(t)*k)
    return B

def bg_hom3d(t: ArrayLike, conc: float, lamb: float) -> np.ndarray:
    """Generates an exponential decay vector from conc and lamb.

    Assumes a homogeneous 3D distribution of background spins.

    Parameters
    ----------
    t : ArrayLike
        The time axis to generate the background decay on.
    conc : float
        The spin concentration in micromolars.
    lamb : float
        The modulation depth.
    
    Returns
    -------
    B : np.ndarray
        The background decay vector.
    """
    conc = conc*1e-6*1e3*NA # umol/L -> mol/L -> mol/m^3 -> spins/m^3
    B = np.exp(-8*m.pi**2/9/m.sqrt(3)*lamb*conc*D*abs(t*1e-6))
    return B