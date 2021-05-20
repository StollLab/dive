import numpy as np
import math as m
from theano import tensor as T
from theano.tensor import nlinalg as tnp
from theano.tensor import slinalg as snp
import pymc3 as pm
from scipy.linalg import sqrtm
import deerlab as dl
import matplotlib.pyplot as plt


from pymc3.step_methods.arraystep import BlockedStep

class SampleEdwardsModel(BlockedStep):
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
        # sigma = np.exp(point[self.sigma.transformed.name])
        sigma = self.sigma
        tau = 1/(sigma**2)
        delta = np.exp(point[self.delta.transformed.name])
       
        new = point.copy()
        new[self.var.name] = randP(delta,tau,self.KtK,self.KtS,self.LtL,self.nr)

        return new

class SampleExpandedEdwardsModel(BlockedStep):
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
        sigma = np.exp(point[self.sigma.transformed.name])
        tau = 1/(sigma**2)
        delta = np.exp(point[self.delta.transformed.name])
        V0 = np.exp(point[self.V0.transformed.name])

        KtS = self.KtS / V0
       
        new = point.copy()
        new[self.var.name] = randP(delta,tau,self.KtK,KtS,self.LtL,self.nr)

        return new

from pymc3.step_methods.arraystep import BlockedStep
import time

class SampleFullP(BlockedStep):
    def __init__(self, var, delta, sigma, k, lamb, V0, LtL, t, V, r):
            self.vars = [var]
            self.var = var
            self.delta = delta
            self.sigma = sigma
            self.k = k
            self.lamb = lamb
            self.LtL = LtL
            self.V0 = V0
            self.V = V
            self.r = r
            self.t = t

    def step(self, point: dict):
        # transform parameters
        sigma = np.exp(point[self.sigma.transformed.name])
        tau = 1/(sigma**2)
        delta = np.exp(point[self.delta.transformed.name])
        # k = np.exp(point[self.k.transformed.name])
        k = 0
        lamb = np.exp(point[self.lamb.transformed.name])
        V0 = np.exp(point[self.V0.transformed.name])

        nr = len(self.r)

        # Background
        B = dl.bg_exp(self.t,k) 
        # Kernel
        K = dl.dipolarkernel(self.t, self.r, mod = lamb, bg = B, integralop=True)
        K[:,0] = 2*K[:,0]
        K[:,-1] = 2*K[:,-1]

        KtK = np.matmul(np.transpose(K),K)
        KtV = np.matmul(np.transpose(K),self.V) / V0

        new = point.copy()
        Pdraw = randP(delta,tau,KtK,KtV,self.LtL,nr)

        # plt.plot(self.r,Pdraw)
        # plt.show()
        new[self.var.name] = Pdraw

        return new


class SampleFullP_(BlockedStep):
    def __init__(self, var, delta, sigma, k, lamb, V0, LtL, t, Vfull, r):
            self.vars = [var]
            self.var = var
            self.delta = delta
            self.sigma = sigma
            self.k = k
            self.lamb = lamb
            self.LtL = LtL
            self.V0 = V0
            self.Vfull = Vfull
            self.r = r
            self.t = t

    def step(self, point: dict):
        # transform parameters
        sigma = np.exp(point[self.sigma.transformed.name])
        tau = 1/(sigma**2)
        delta = np.exp(point[self.delta.transformed.name])
        k = np.exp(point[self.k.transformed.name])
        lamb = np.exp(point[self.lamb.transformed.name])
        V0 = np.exp(point[self.V0.transformed.name])

        nr = len(self.r)

        # Background
        B = dl.bg_exp(self.t,k) 
        # Kernel
        K = dl.dipolarkernel(self.t, self.r, integralop=True)
        K[:,0] = 2*K[:,0]
        K[:,-1] = 2*K[:,-1]

        V_ = np.divide(self.Vfull/V0,B)
        S = (V_-1+lamb)/lamb
        
        KtK = np.matmul(np.transpose(K),K)
        KtS = np.matmul(np.transpose(K),S)

        new = point.copy()
        Pdraw = randP(delta,tau,KtK,KtS,self.LtL,nr)

        # plt.plot(self.r,Pdraw)
        # plt.show()
        new[self.var.name] = Pdraw

        return new

def randP(delta,tau,KtK,KtS,LtL,nr):
    r"""
    based on:
    J.M. Bardsley, C. Fox, An MCMC method for uncertainty quantification in
    nonnegativity constrained inverse problems, Inverse Probl. Sci. Eng. 20 (2012)
    """
    invSigma = tau*KtK + delta*LtL
    Sigma = np.linalg.inv(invSigma)

    try:
        #'lower' syntax is faster for sparse matrices. Also matches convention in
        # Bardsley paper.
        C_L = np.linalg.cholesky(Sigma)
    except:
        C_L = sqrtm(Sigma)
        
    v = np.random.standard_normal(size=(nr,))
    w = np.linalg.lstsq(np.matrix.transpose(C_L),v,rcond=None)
    w = w[0]

    P = fnnls(invSigma,tau*KtS+w)

    return P

def fnnls(AtA,Atb,tol=[],maxiter=[],verbose=False):
#=====================================================================================
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
            x_[passive] = np.linalg.solve(AtA[np.ix_(passive,passive)],Atb[passive])
        
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
            print('{:10.0f}{:15.0f}{:20.4e}\n'.format(outIteration,iIteration,max(w)) )

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

#=====================================================================================
