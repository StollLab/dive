from dataclasses import replace
import numpy as np
import math as m
import pandas as pd
from pandas.core import indexers
from scipy.special import fresnel

from .constants import *
from .deerload import *

import arviz as az
from .plotting import *

from typing import Union
from numpy.typing import ArrayLike

def addnoise(V: ArrayLike, sig: float) -> ArrayLike:
    """Adds Gaussian noise with standard deviation sig to signal.

    Parameters
    ----------
    V : ArrayLike
        The signal to add noise to.
    sig : float
        The standard deviation of the noise to add.
    
    Returns
    -------
    Vnoisy : ArrayLike

    See Also
    --------
    np.random.normal
    """
    noise = np.random.normal(0, sig, np.size(V))
    Vnoisy = V + noise
    return Vnoisy


def FWHM2sigma(FWHM: float) -> float:
    """Converts full width at half maximum (FWHM) to standard deviation.

    Parameters
    ----------
    FWHM : float
        The full width at half maximum of a Gaussian.
    
    Returns
    -------
    sigma : float
        The standard deviation of the Gaussian.
    """
    sigma = FWHM/(2*m.sqrt(2*m.log(2)))
    return sigma


def sigma2FWHM(sigma) -> float:
    """Converts standard deviation to full width at half maximum (FWHM)

    Parameters
    ----------
    sigma : float
        The standard deviation of a Gaussian
    
    Returns
    -------
    FWHM : float
        The full width at half maximum of the Gaussian.
    """
    FWHM = sigma/(2*m.sqrt(2*m.log(2)))
    return FWHM


def dipolarkernel(t: ArrayLike, r: ArrayLike) -> np.ndarray:
    """Calculates the dipolar kernel matrix.

    Generates given a time vector t in microseconds and a distance
    vector r in nanometers.

    Parameters
    ----------
    t : ArrayLike
        The time vector, in microseconds.
    r : ArrayLike
        The distance vector, in nanometers.

    Returns
    -------
    K : np.ndarray
        The dipolar kernel matrix.
    """
    omega = 1e-6 * D/(r*1e-9)**3  # rad µs^-1
    
    # Calculation using Fresnel integrals
    nr = np.size(r)
    nt = np.size(t)
    K = np.zeros((nt, nr))
    for ir in range(nr):
        ph = omega[ir]*np.abs(t)
        z = np.sqrt(6*ph/m.pi)
        S, C = fresnel(z)
        K[:,ir] = (C*np.cos(ph)+S*np.sin(ph))/z
    
    K[t==0,:] = 1  # fix div by zero
    
    # Include delta-r factor for integration
    if len(r)>1:
        dr = np.mean(np.diff(r))
        K *= dr
    
    return K

def get_rhats(trace: az.InferenceData) -> pd.Series:
    """Gets the r-hat statistics for each variable in a trace.

    R-hat is the ratio of interchain to intrachain variance and is a
    good indicator of convergence. When R-hat is close to 1, the chains
    are converged.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be analyzed.
    
    Returns
    -------
    rhats : pd.Series
        A pd.Series object containing the variable names and their r-hat
        values.
    
    See Also
    --------
    prune_chains
    az.summary
    """
    rhats = az.summary(trace, var_names=["~P"], filter_vars="like")["r_hat"]
    return rhats

def prune_chains(
    trace: az.InferenceData, max_remove: int = None, max_allowed: float = 1.05, 
    min_change: float = 0, return_chain_nums: bool = False, depth: int = 0, 
    chain_nums: list[int] = None) -> Union[az.InferenceData, list[int]]:
    """Prunes chains from a trace to attain convergence.

    The algorithm drops chains one by one and iteratively removes the
    chain that reduces maximum r-hat the most when dropped. This process
    continues until the maximum r-hat is less than max_allowed. This 
    only removes up to max_remove of the chains, and only if removing 
    the chain reduces maximum r-hat by more than min_change.

    Returns the trace with clean chains, unless return_chain_nums is
    set to True.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be pruned.
    max_remove : int, optional
        The maximum number of chains to be removed. Defaults to half 
        the number of chains in the trace.
    max_allowed : float, default=1.05
        The maximum allowable value of r-hat. When reached, the function 
        will stop pruning.
    min_change : float, default=0
        The minimum reduction in r-hat for a chain to be pruned.
    return_chain_nums : bool, default=False
        Whether or not to return the numbers of the chains to be kept, 
        instead of the pruned trace.
    depth : int, default=0
        Internal parameter to keep track of recursion.
    chain_nums : list of int, optional
        Internal parameter to keep track of remaining chains.

    Returns
    -------
    to_return : az.InferenceData or list of int
        Either the pruned trace or the list of chain numbers to keep,
        depending on the value of return_chain_nums.
    
    See Also
    --------
    get_rhats
    az.sel
    """
    # get chain numbers
    if chain_nums is None:
        chain_nums = [j for j in range(trace.posterior.dims["chain"])]

    # by default only removes up to half the chains
    if max_remove is None:
        max_remove = int(trace.posterior.dims["chain"]/2)

    # get rhats
    rhats = get_rhats(trace.sel(chain=chain_nums))
    rhat_max, rhat_min = rhats.max(), rhats.min()
    rhat_spread = rhat_max - rhat_min

    # break if break criterion met
    exit = False
    if rhat_max < max_allowed:
        exit = True
    if depth >= max_remove:
        exit = True

    if exit:
        if return_chain_nums:
            to_return = chain_nums
        else:
            to_return = trace.sel(chain=chain_nums)
    else:
        # finds the variable with the highest rhat
        # then drops each chain one by one and see which one 
        # minimizes rhat in that variable
        to_drop = None
        idx_of_rhat_max = rhats.idxmax()
        lowest_rhat_max = rhats[idx_of_rhat_max]
        for i in chain_nums:
            chain_nums_copy = chain_nums.copy()
            chain_nums_copy.remove(i)
            dropped_rhats = get_rhats(trace.sel(chain=chain_nums_copy))
            if dropped_rhats[idx_of_rhat_max] < lowest_rhat_max-min_change:
                lowest_rhat_max = dropped_rhats[idx_of_rhat_max]
                to_drop = i
        # update chain_nums with the chain to drop and recurse the function
        if to_drop is not None:
            chain_nums.remove(to_drop)
            depth += 1
        else:
            exit = True
        to_return = prune_chains(trace, max_remove, max_allowed, min_change, 
                                 return_chain_nums, depth, chain_nums)
    return to_return

def fnnls(
    AtA: ArrayLike, Atb: ArrayLike, tol: float = None, maxiter: int = None, 
    verbose: bool = False):
    r"""Fast non-negative least-squares algorithm.

    x = fnnls(AtA,Atb) solves the problem min ||b - Ax|| if
    AtA = A'*A and Atb = A'*b.
    A default tolerance of TOL = MAX(SIZE(AtA)) * NORM(AtA,1) * EPS
    is used for deciding when elements of x are less than zero.
    This can be overridden with x = fnnls(AtA,Atb,TOL).

    [x,w] = fnnls(AtA,Atb) also returns dual vector w where
    w(i) < 0 where x(i) = 0 and w(i) = 0 where x(i) > 0.

    Parameters
    ----------
    AtA, Atb : ArrayLike
        The matrices/vectors to use in the FNNLS algorithm.
    tol : float, optional
        The tolerance. If not provided, it will be automatically
        calculated.
    maxiter : int, optional
        The maximum number of iterations. If not provided, it will
        be automatically calculated.
    verbose : bool, default=False
        Whether to print the status of solution-finding.
    
    Returns
    -------
    x : np.ndarray
        The solution to the optimization problem.

    References
    ----------
    .. [1] R. Bro, S. De Jong
       A Fast Non-Negativity-Constrained Least Squares Algorithm
       Journal of Chemometrics 11 (1997) 393-401
    .. [2] Lawson and Hanson, "Solving Least Squares Problems", 
       Prentice-Hall, 1974.
    """

    if np.iscomplex(AtA).any():
        print('FNNLS called with complex-valued AtA.')
    if np.iscomplex(Atb).any():
        print('FNNLS called with complex-valued Atb.')

    unsolvable = False
    count = 0

    # Use all-zero starting vector
    N = np.shape(AtA)[1]
    x = np.zeros(N)

    # Calculate tolerance and maxiter if not given.
    if tol is None:
        eps = np.finfo(float).eps
        tol = 10*eps*np.linalg.norm(AtA,1)*max(np.shape(AtA))
    if maxiter is None:
        maxiter = 5*N

    # initial positive/passive set (points where constraint is not active)
    passive = x>0
    x[~passive] = 0
    # negative gradient of error functional 0.5*||A*x-y||^2
    w = Atb - AtA @ x     
    
    # Outer loop: 
    # Add variables to positive set if w indicates that fit can be improved.
    outIteration = 0
    maxIterations = 5*N
    while np.any(w>tol) and np.any(~passive):
        outIteration += 1
        # Add the most promising variable (with largest w) to positive set.
        t = np.argmax(w)
        passive[t] = True
        # Solve unconstrained problem for new augmented positive set.
        # Gives a candidate solution with potentially new negative variables.
        x_ = np.zeros(N)
        if np.sum(passive)==1:
            x_[passive] = Atb[passive]/AtA[passive,passive]
        else:
            x_[passive] = np.linalg.solve(AtA[np.ix_(passive,passive)], 
                                          Atb[passive])
        
        # Inner loop: 
        # Iteratively eliminate negative variables from candidate solution.
        iIteration = 0
        while any((x_<=tol) & passive) and iIteration<maxIterations:
            iIteration += 1
            # Calculate maximum feasible step size and do step.
            negative = (x_<=tol) & passive
            alpha = min(x[negative]/(x[negative]-x_[negative]))
            x += alpha*(x_-x)
            # Remove all negative variables from positive set.
            passive[x<tol] = False
            # Solve unconstrained problem for reduced positive set.
            x_ = np.zeros(N)
            if np.sum(passive)==1:
                x_[passive] = Atb[passive]/AtA[passive,passive]
            else:
                x_[passive] = np.linalg.solve(AtA[np.ix_(passive,passive)],
                                              Atb[passive])
            
        # Accept non-negative candidate solution and calculate w.
        if all(x == x_):
            count += 1
        else:
            count = 0
        if count > 5:
            unsolvable = True
            break
        x = x_
        
        w = Atb - AtA@x
        w[passive] = -m.inf
        if verbose:
            print(f"{outIteration:10.0f}{iIteration:15.0f}{max(w):20.4e}\n")

    # Print status of optimization
    if verbose:
        if unsolvable:
            print('Optimization stopped because the solution cannot be further changed. \n')
        elif any(~passive):
            print('Optimization stopped because the active set has been completely emptied. \n')
        elif w>tol:
            print('Optimization stopped because the gradient (w) is inferior than the tolerance value TolFun = #.6e. \n' %tol)
        else:
            print('Solution found. \n')
    
    return x
