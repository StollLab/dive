import numpy as np
import math as m
from scipy.linalg import sqrtm
import deerlab as dl
from pymc3.step_methods.arraystep import BlockedStep
import pymc3 as pm
import scipy as sp

from pymc3.distributions.transforms import log




from .deer import *

class randP_EdwardsModel(BlockedStep):
    def __init__(self, var, delta, sigma, KtK, KtS, LtL, nr):
            self.vars = [var]
            self.var = var
            self.delta = delta
            self.sigma = sigma
            self.KtK = KtK
            self.KtS = KtS
            self.LtL = LtL
            self.nr = nr

    def step(self, point: dict):
        
        # Get model parameters
        sigma = self.sigma
        tau = 1/(sigma**2)
        delta = undo_transform(point, self.delta)
       
        # Calculate distribution parameters 
        tauKtS = tau * self.KtS
        invSigma = tau*self.KtK + delta*self.LtL
        
        # Draw new sample of P
        Pdraw = _randP(tauKtS,invSigma)
        
        # Save new sample
        newpoint = point.copy()
        newpoint[self.var.name] = Pdraw

        return newpoint

class randP_ExpandedEdwardsModel(BlockedStep):
    def __init__(self, var, delta, sigma, V0, KtK, KtS, LtL, nr):
            self.vars = [var]
            self.var = var
            self.delta = delta
            self.sigma = sigma
            self.V0 = V0
            self.KtK = KtK
            self.KtS = KtS
            self.LtL = LtL
            self.nr = nr

    def step(self, point: dict):
        
        # Get model parameters
        sigma = undo_transform(point, self.sigma)
        tau = 1/(sigma**2)
        delta = undo_transform(point, self.delta)
        V0 = undo_transform(point, self.V0)

        # Calculate distribution parameters
        tauKtS = tau * self.KtS / V0
        invSigma = tau*self.KtK + delta*self.LtL
        
        # Draw new sample of P
        Pdraw = _randP(tauKtS, invSigma)
        
        # Save new sample
        newpoint = point.copy()
        newpoint[self.var.name] = Pdraw

        return newpoint

class randPnorm_k_posterior(BlockedStep):
    def __init__(self, var, K0, LtL, t, V, r, delta, sigma, tau, k, lamb, V0):
            self.vars = [var]
            self.var = var
            
            # precalculated values
            self.K0 = K0
            self.LtL = LtL
            self.V = V
            self.t = t
            self.dr = r[1]-r[0]

            # random variables
            self.delta = delta
            self.sigma = sigma
            self.k = k
            self.lamb = lamb
            self.V0 = V0
            self.tau = tau  

    def step(self, point: dict):
        # Get parameters
        tau = undo_transform(point, self.tau)
        delta = undo_transform(point, self.delta)
        k = undo_transform(point, self.k)
        lamb = undo_transform(point, self.lamb)
        V0 = undo_transform(point, self.V0) 

        # Calculate kernel matrix
        K = (1-lamb) + lamb*self.K0
        B = bg_exp(self.t,k)  
        K *= B[:, np.newaxis]
        K *= V0*self.dr

        # Calculate distribution parameters
        KtK = np.matmul(np.transpose(K), K)
        KtV = np.matmul(np.transpose(K), self.V) 
        tauKtV = tau*KtV
        invSigma = tau*KtK + delta*self.LtL
        
        # Draw new sample of P and normalize
        Pdraw = _randP(tauKtV, invSigma)
        Pdraw =  Pdraw / np.sum(Pdraw) / self.dr
        
        # Store new sample
        newpoint = point.copy()
        newpoint[self.var.name] = Pdraw

        return newpoint

class randPnorm_tauB_posterior(BlockedStep):
    def __init__(self, var, K0, LtL, t, V, r, delta, sigma, tau, tauB, lamb, V0):
            self.vars = [var]
            self.var = var
            
            # precalculated values
            self.K0 = K0
            self.LtL = LtL
            self.V = V
            self.t = t
            self.dr = r[1]-r[0]

            # random variables
            self.delta = delta
            self.sigma = sigma
            self.tauB = tauB
            self.lamb = lamb
            self.V0 = V0
            self.tau = tau  

    def step(self, point: dict):
        # Get parameters
        tau = undo_transform(point, self.tau)
        delta = undo_transform(point, self.delta)
        tauB = undo_transform(point, self.tauB)
        lamb = undo_transform(point, self.lamb)
        V0 = undo_transform(point, self.V0) 

        # Calculate kernel matrix
        K = (1-lamb) + lamb*self.K0
        B = bg_exp_time(self.t,tauB) 
        K *= B[:, np.newaxis]
        K *= V0*self.dr

        # Calculate distribution parameters
        KtK = np.matmul(np.transpose(K), K)
        KtV = np.matmul(np.transpose(K), self.V) 
        tauKtV = tau*KtV
        invSigma = tau*KtK + delta*self.LtL
        
        # Draw new sample of P and normalize
        Pdraw = _randP(tauKtV, invSigma)
        Pdraw =  Pdraw / np.sum(Pdraw) / self.dr
        
        # Store new sample
        newpoint = point.copy()
        newpoint[self.var.name] = Pdraw

        return newpoint

class randPnorm_Bend_posterior(BlockedStep):
    def __init__(self, var, K0, LtL, t, V, r, delta, sigma, tau, Bend, lamb, V0):
            self.vars = [var]
            self.var = var
            
            # precalculated values
            self.K0 = K0
            self.LtL = LtL
            self.V = V
            self.t = t
            self.dr = r[1]-r[0]

            # random variables
            self.delta = delta
            self.sigma = sigma
            self.Bend = Bend
            self.lamb = lamb
            self.V0 = V0
            self.tau = tau  

    def step(self, point: dict):
        # Get parameters
        tau = undo_transform(point, self.tau)
        delta = undo_transform(point, self.delta)
        Bend = undo_transform(point, self.Bend)
        lamb = undo_transform(point, self.lamb)
        V0 = undo_transform(point, self.V0) 

        # Calculate kernel matrix
        K = (1-lamb) + lamb*self.K0
        #k = (1/np.max(self.t))*np.log((1-lamb)/Bend)
        k = -1/self.t[-1]*np.log(Bend)
        B = bg_exp(self.t,k) 
        K *= B[:, np.newaxis]
        K *= V0*self.dr

        # Calculate distribution parameters
        KtK = np.matmul(np.transpose(K), K)
        KtV = np.matmul(np.transpose(K), self.V) 
        tauKtV = tau*KtV
        invSigma = tau*KtK + delta*self.LtL
        
        # Draw new sample of P and normalize
        Pdraw = _randP(tauKtV, invSigma)
        Pdraw =  Pdraw / np.sum(Pdraw) / self.dr
        
        # Store new sample
        newpoint = point.copy()
        newpoint[self.var.name] = Pdraw

        return newpoint


class randDelta_posterior(BlockedStep):
    
    def __init__(self, var, delta_prior, L, P):
            self.vars = [var]
            self.var = var
            
            # constants
            self.a_delta = delta_prior[0]
            self.b_delta = delta_prior[1]
            self.L = L
            
            # random variables
            self.P = P

    def step(self, point: dict):
        
        # Get parameters
        P = undo_transform(point, self.P)

        # Calculate distribution parameters
        n_p = sum(np.asarray(P)>0)
        a_ = self.a_delta + n_p/2
        b_ = self.b_delta + (1/2)*np.linalg.norm(self.L@P)**2

        # Draw new sample of delta
        delta_draw = np.random.gamma(a_, 1/b_)
        
        # Save sample
        newpoint = point.copy()
        newpoint[self.var.name] = delta_draw
        
        return newpoint

class randTau_tauB_posterior(BlockedStep):
    """
    based on:
    J.M. Bardsley, P.C. Hansen, MCMC Algorithms for Computational UQ of 
    Nonnegativity Constrained Linear Inverse Problems, 
    SIAM Journal on Scientific Computing 42 (2020) A1269-A1288 
    from "Hierarchical Gibbs Sampler" block after Eqn. (2.8)
    """
    def __init__(self, var, tau_prior, K0, P, V, r, t, tauB, lamb, V0):
            self.vars = [var]
            self.var = var
            
            # data
            self.V = V
            self.t = t
            
            # constants
            self.a_tau = tau_prior[0]
            self.b_tau = tau_prior[1]
            self.K0dr = K0*(r[1]-r[0])
            
            # random variables
            self.P = P
            self.tauB = tauB
            self.lamb = lamb
            self.V0 = V0

    def step(self, point: dict):
        
        # Get parameters
        P = undo_transform(point, self.P)
        tauB = undo_transform(point, self.tauB)
        lamb = undo_transform(point, self.lamb)
        V0 = undo_transform(point, self.V0)  

        # Calculate kernel matrix
        Vmodel = self.K0dr@P
        Vmodel = (1-lamb) + lamb*Vmodel
        B = bg_exp_time(self.t, tauB) 
        Vmodel *= B
        Vmodel *= V0
        
        # Calculate distribution parameters
        M = len(self.V)
        a_ = self.a_tau + M/2
        b_ = self.b_tau + (1/2)*np.linalg.norm((Vmodel-self.V))**2

        # Draw new sample of tau
        tau_draw = np.random.gamma(a_, 1/b_)

        # Save new sample
        newpoint = point.copy()
        newpoint[self.var.name] = tau_draw

        return newpoint

class randTau_Bend_posterior(BlockedStep):
    """
    based on:
    J.M. Bardsley, P.C. Hansen, MCMC Algorithms for Computational UQ of 
    Nonnegativity Constrained Linear Inverse Problems, 
    SIAM Journal on Scientific Computing 42 (2020) A1269-A1288 
    from "Hierarchical Gibbs Sampler" block after Eqn. (2.8)
    """
    def __init__(self, var, tau_prior, K0, P, V, r, t, Bend, lamb, V0):
            self.vars = [var]
            self.var = var
            
            # data
            self.V = V
            self.t = t
            
            # constants
            self.a_tau = tau_prior[0]
            self.b_tau = tau_prior[1]
            self.K0dr = K0*(r[1]-r[0])
            
            # random variables
            self.P = P
            self.Bend = Bend
            self.lamb = lamb
            self.V0 = V0

    def step(self, point: dict):
        
        # Get parameters
        P = undo_transform(point, self.P)
        Bend = undo_transform(point, self.Bend)
        lamb = undo_transform(point, self.lamb)
        V0 = undo_transform(point, self.V0)  

        # Calculate kernel matrix
        Vmodel = self.K0dr@P
        Vmodel = (1-lamb) + lamb*Vmodel
        k = -1/self.t[-1]*np.log(Bend)
        B = bg_exp(self.t, k) 
        Vmodel *= B
        Vmodel *= V0
        
        # Calculate distribution parameters
        M = len(self.V)
        a_ = self.a_tau + M/2
        b_ = self.b_tau + (1/2)*np.linalg.norm((Vmodel-self.V))**2

        # Draw new sample of tau
        tau_draw = np.random.gamma(a_, 1/b_)

        # Save new sample
        newpoint = point.copy()
        newpoint[self.var.name] = tau_draw

        return newpoint

class randTau_k_posterior(BlockedStep):
    """
    based on:
    J.M. Bardsley, P.C. Hansen, MCMC Algorithms for Computational UQ of 
    Nonnegativity Constrained Linear Inverse Problems, 
    SIAM Journal on Scientific Computing 42 (2020) A1269-A1288 
    from "Hierarchical Gibbs Sampler" block after Eqn. (2.8)
    """
    def __init__(self, var, tau_prior, K0, P, V, r, t, k, lamb, V0):
            self.vars = [var]
            self.var = var
            
            # data
            self.V = V
            self.t = t
            
            # constants
            self.a_tau = tau_prior[0]
            self.b_tau = tau_prior[1]
            self.K0dr = K0*(r[1]-r[0])
            
            # random variables
            self.P = P
            self.k = k
            self.lamb = lamb
            self.V0 = V0

    def step(self, point: dict):
        
        # Get parameters
        P = undo_transform(point, self.P)
        k = undo_transform(point, self.k)
        lamb = undo_transform(point, self.lamb)
        V0 = undo_transform(point, self.V0)  

        # Calculate kernel matrix
        Vmodel = self.K0dr@P
        Vmodel = (1-lamb) + lamb*Vmodel
        B = bg_exp(self.t, k) 
        Vmodel *= B
        Vmodel *= V0
        
        # Calculate distribution parameters
        M = len(self.V)
        a_ = self.a_tau + M/2
        b_ = self.b_tau + (1/2)*np.linalg.norm((Vmodel-self.V))**2

        # Draw new sample of tau
        tau_draw = np.random.gamma(a_, 1/b_)

        # Save new sample
        newpoint = point.copy()
        newpoint[self.var.name] = tau_draw

        return newpoint

def undo_transform(point, rv):
    '''
    Automatically transforms transformed random variables
    (log, logodds, etc) back to their original scale.
    '''
    
    # Don't untransform if variable is not transformed
    if isinstance(rv, pm.model.FreeRV):
        value = point[rv.name]
        return value

    key = rv.transformed.name
    transform_marker = key.split('_')[1]
    value = point[key]
    
    if transform_marker == 'log':
        return np.exp(value)
    elif transform_marker == 'lowerbound':
        return np.exp(value)
    elif transform_marker == 'logodds':
        return sp.special.expit(value)
    elif transform_marker == 'interval':
        return sp.special.expit(value)
    else:
        raise ValueError('Could not figure out RV transformation.')

def _randP(tauKtX,invSigma):
    r"""
    based on:
    J.M. Bardsley, C. Fox, An MCMC method for uncertainty quantification in
    nonnegativity constrained inverse problems, Inverse Probl. Sci. Eng. 20 (2012)
    """
    Sigma = np.linalg.inv(invSigma)

    try:
        C_L = np.linalg.cholesky(Sigma)
    except:
        C_L = sqrtm(Sigma)
        
    v = np.random.standard_normal(size=(len(tauKtX),))
    w = np.linalg.solve(np.matrix.transpose(C_L), v)
    
    P = fnnls(invSigma, tauKtX+w)
    
    return P

def fnnls(AtA, Atb, tol=[], maxiter=[], verbose=False):
    r"""
    FNNLS   Fast non-negative least-squares algorithm.
    x = fnnls(AtA,Atb) solves the problem min ||b - Ax|| if
        AtA = A'*A and Atb = A'*b.
    A default tolerance of TOL = MAX(SIZE(AtA)) * NORM(AtA,1) * EPS
    is used for deciding when elements of x are less than zero.
    This can be overridden with x = fnnls(AtA,Atb,TOL).

    [x,w] = fnnls(AtA,Atb) also returns dual vector w where
        w(i) < 0 where x(i) = 0 and w(i) = 0 where x(i) > 0.
    
    For the FNNLS algorithm, see
        R. Bro, S. De Jong
        A Fast Non-Negativity-Constrained Least Squares Algorithm
        Journal of Chemometrics 11 (1997) 393-401
    The algorithm FNNLS is based on is from
        Lawson and Hanson, "Solving Least Squares Problems", Prentice-Hall, 1974.
    """

    unsolvable = False
    count = 0

    # Use all-zero starting vector
    N = np.shape(AtA)[1]

    x = np.zeros(N)

    # Calculate tolerance and maxiter if not given.
    if np.size(np.atleast_1d(tol))==0:
        eps = np.finfo(float).eps
        tol = 10*eps*np.linalg.norm(AtA,1)*max(np.shape(AtA))
    if np.size(np.atleast_1d(maxiter))==0:
        maxiter = 5*N


    passive = x>0       # initial positive/passive set (points where constraint is not active)
    x[~passive] = 0
    w = Atb - AtA @ x     # negative gradient of error functional 0.5*||A*x-y||^2
    
    # Outer loop: Add variables to positive set if w indicates that fit can be improved.
    outIteration = 0
    maxIterations = 5*N    
    while np.any(w>tol) and np.any(~passive):
        outIteration += 1
        
        # Add the most promising variable (with largest w) to positive set.
        t = np.argmax(w)
        passive[t] = True
        
        # Solve unconstrained problem for new augmented positive set.
        # This gives a candidate solution with potentially new negative variables.
        x_ = np.zeros(N)
        
        if any( Atb == complex()):
            print('Áhh, complex')

        if np.sum(passive)==1:
            x_[passive] = Atb[passive]/AtA[passive,passive]
        else:
            x_[passive] = np.linalg.solve(AtA[np.ix_(passive,passive)], Atb[passive])
        
        # Inner loop: Iteratively eliminate negative variables from candidate solution.
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
                x_[passive] = np.linalg.solve(AtA[np.ix_(passive,passive)],Atb[passive])
            
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