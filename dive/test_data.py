import deerlab as dl
import numpy as np

from .deer import *

def generate_single_gauss(
    r0: float = 4, w: float = 0.4, lamb: float = 0.5, k: float = 0.1, 
    V0: float = 1, sigma: float = 0.01, seed: int = 0, nr: int = 800, 
    nt: int = 150, r_lim: tuple[float,float] = [1,10], 
    t_lim: tuple[float,float] = [-0.1,2.5]) -> tuple[dict,dict]:
    """Generates a single-gauss P(r) and associated V(t).

    Parameters
    ----------
    r0 : float, default=4
        The mean of the Gaussian.
    w : float, default=0.4
        The full width at half-maximum of the Gaussian.
    lamb : float, default=0.5
        The modulation depth.
    k : float, default=0.1
        The background decay rate.
    V0 : float, default=1
        The signal amplitude.
    sigma : float, default=0.01
        The noise level.
    seed : int, default=0
        The random seed to use.
    nr : int, default=800
        The number of points to use for the distance axis.
    nt : int, default=150
        The number of points to use for the time axis.
    r_lim : tuple of float, float
        The minimum and maximum values of the distance axis.
    t_lim : tuple of float, float
        The minimum and maximum values of the time axis.
    
    Returns
    -------
    (data, pars) : tuple of dict, dict
        The `data` dictionary contains the P(r) and V(t) data.
        The 'pars' dictionary stores the parameters used for data 
        generation.
    
    See Also
    --------
    dl.dd_gauss
    """
    t = np.linspace(t_lim[0],t_lim[1],nt)        # time axis, µs
    r = np.linspace(r_lim[0],r_lim[1],nr)      # distance axis, nm

    P = dd_gauss(r,r0,w)          # model distance distribution
    B = dl.bg_exp(t,k)         # background decay

    K = dl.dipolarkernel(t,r,integralop=True)    # kernel matrix
    K[:,0] = 2*K[:,0]
    K[:,-1] = 2*K[:,-1]
    KB = dl.dipolarkernel(t,r,mod=lamb,bg=B,integralop=True)
    KB[:,0] = 2*KB[:,0]
    KB[:,-1] = 2*KB[:,-1]

    Sm = K@P
    S = Sm + dl.whitegaussnoise(t,sigma,seed = seed)
    Vm = V0*KB@P
    V = Vm + dl.whitegaussnoise(t,sigma,seed = seed)

    pars = {'gaussian': [r0,w], 'lamb': lamb, 'k': k, 'V0': V0, 'sigma': sigma, 
            'seed': seed}
    data = {'t': t, 'V': V, 'S': S, 'r': r, 'P': P, 'V0': Vm, 'S0': Sm}
    return data, pars


def generate_multi_gauss(
    r0: tuple[float,...] = [4,4.8], w: tuple[float,...] = [0.6,1.2], 
    a: tuple[float,...] = [0.6,0.4], lamb: float = 0.5, k: float = 0.1, 
    V0: float = 1, sigma: float = 0.01, seed: int = 0, nr: int = 800, 
    nt: int = 150, r_lim: tuple[float,float] = [1,10], 
    t_lim: tuple[float,float] = [-0.1,2.5]) -> tuple[dict,dict]:
    """Generates a multi-gauss P(r) and associated V(t).

    Parameters
    ----------
    r0 : tuple of float, default=[4,4.8]
        The means of the Gaussians.
    w : tuple of flaot, default=[0.3,0.6]
        The full widths at half maximum of the Gaussians.
    a : tuple of float, default=[0.6,0.4]
        The amplitudes of the Gaussians.
    lamb : float, default=0.5
        The modulation depth.
    k : float, default=0.1
        The background decay rate.
    V0 : float, default=1
        The signal amplitude.
    sigma : float, default=0.01
        The noise level.
    seed : int, default=0
        The random seed to use.
    nr : int, default=800
        The number of points to use for the distance axis.
    nt : int, default=150
        The number of points to use for the time axis.
    r_lim : tuple of float, float
        The minimum and maximum values of the distance axis.
    t_lim : tuple of float, float
        The minimum and maximum values of the time axis.
    
    Returns
    -------
    (data, pars) : tuple of dict, dict
        The `data` dictionary contains the P(r) and V(t) data.
        The 'pars' dictionary stores the parameters used for data 
        generation.
    
    See Also
    --------
    dl.dd_gauss2
    """
    t = np.linspace(t_lim[0],t_lim[1],nt)        # time axis, µs
    r = np.linspace(r_lim[0],r_lim[1],nr)      # distance axis, nm

    P = dd_gauss(r,r0,w,a)          # model distance distribution
    B = dl.bg_exp(t,k)         # background decay

    K = dl.dipolarkernel(t,r,integralop=True)    # kernel matrix
    K[:,0] = 2*K[:,0]
    K[:,-1] = 2*K[:,-1]
    KB = dl.dipolarkernel(t,r,mod=lamb,bg=B,integralop=True)
    KB[:,0] = 2*KB[:,0]
    KB[:,-1] = 2*KB[:,-1]

    Sm = K@P
    S = Sm + dl.whitegaussnoise(t,sigma,seed = seed)
    Vm = V0*KB@P
    V = Vm + + dl.whitegaussnoise(t,sigma,seed = seed)

    pars = {'r0': r0, 'w': w, 'a': a, 'lamb': lamb, 'k': k, 'V0': V0, 
            'sigma': sigma, 'seed': seed}
    data = {'t': t, 'V': V, 'S': S, 'r': r, 'P': P, 'V0': Vm, 'S0': Sm}
    return data, pars