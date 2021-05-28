import numpy as np
import math as m
from scipy.linalg import sqrtm
import deerlab as dl
from pymc3.step_methods.arraystep import BlockedStep
import pymc3 as pm
import scipy as sp
from pymc3.distributions.transforms import log

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
        sigma = self.sigma
        tau = 1/(sigma**2)
        delta = undo_transform(point,self.delta.transformed.name)
       
        tauKtS = tau * self.KtS
        invSigma = tau*self.KtK + delta*self.LtL
        
        newpoint = point.copy()
        Pdraw = randP(tauKtS,invSigma)
        newpoint[self.var.name] = Pdraw

        return newpoint

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
        sigma = undo_transform(point,self.sigma.transformed.name)
        tau = 1/(sigma**2)
        delta = undo_transform(point,self.delta.transformed.name)
        V0 = undo_transform(point,self.V0.transformed.name)

        tauKtS = tau * self.KtS / V0
        invSigma = tau*self.KtK + delta*self.LtL
        
        newpoint = point.copy()
        Pdraw = randP(tauKtS,invSigma)
        newpoint[self.var.name] = Pdraw

        return newpoint

class SamplePfromV(BlockedStep):
    def __init__(self, var, K0, LtL, t, V, r, delta, sigma, k, lamb, V0):
            self.vars = [var]
            self.var = var
            
            # precalculated values
            self.K0 = K0
            self.LtL = LtL
            self.V = V
            self.r = r
            self.t = t

            # random variables
            self.delta = delta
            self.sigma = sigma
            self.k = k
            self.lamb = lamb
            self.V0 = V0    

    def step(self, point: dict):
        # transform parameters
        sigma = undo_transform(point,self.sigma.transformed.name)
        # delta = undo_transform(point,self.delta.transformed.name)
        delta = undo_transform(point,self.delta.name)
        k = undo_transform(point,self.k.transformed.name)
        lamb = undo_transform(point,self.lamb.transformed.name)
        V0 = undo_transform(point,self.V0.transformed.name)        

        # calculate some values
        tau = 1/(sigma**2)
        dr = self.r[1] - self.r[0]

        # Background
        B = dl.bg_exp(self.t,k) 

        # Kernel
        Kintra = (1-lamb)+lamb*self.K0
        K = Kintra * B[:, np.newaxis]
        K = V0*K*dr

        # KtXs
        KtK = np.matmul(np.transpose(K),K)
        KtV = np.matmul(np.transpose(K),self.V) 

        tauKtV = tau*KtV
        invSigma = tau*KtK + delta*self.LtL
        
        newpoint = point.copy()
        Pdraw = randP(tauKtV,invSigma)
        Pdraw =  Pdraw / np.sum(Pdraw) / dr
        newpoint[self.var.name] = Pdraw

        return newpoint

class randDelta(BlockedStep):
    def __init__(self, var, a0, b0, L, P):
            self.vars = [var]
            self.var = var
            self.P = P
            self.a0 = a0
            self.b0 = b0
            self.L = L

    def step(self, point: dict):
        P = point[self.P.name]

        nr = len(P)
        a_ = self.a0 + nr/2
        b_ = self.b0 + (1/2)*np.linalg.norm(self.L@P)**2

        newpoint = point.copy()
        delta_draw =  np.random.gamma(a_, 1/b_, 1)[0]
        newpoint[self.var.name] = delta_draw

        return newpoint

class randDelta2(BlockedStep):
    def __init__(self, var, a0, b0):
            self.vars = [var]
            self.var = var
            self.a0 = a0
            self.b0 = b0


    def step(self, point: dict):
        newpoint = point.copy()
        # print(newpoint[self.var.transformed.name])
        delta_draw =  np.random.gamma(self.a0, 1/self.b0, 1)[0]
        # newpoint[self.var.transformed.name] = log.forward_val(delta_draw)
        newpoint[self.var.name] = 2.
        # newpoint[self.var.name] = 2.
        # print(newpoint[self.var.transformed.name])

        return newpoint


def undo_transform(point,key):
    '''
    Automatically transforms variables which were sampled on log 
    or logodds scale back into original scale.
    '''
    x = point[key]

    try:
        transform_marker = key.split('_')[1]
     
        if transform_marker == 'log' or transform_marker == 'lowerbound':
            y = np.exp(x)
            
        elif transform_marker == 'logodds':
            y = sp.special.expit(x)

        return y

    except:
        return x

def randP(tauKtX,invSigma):
    r"""
    based on:
    J.M. Bardsley, C. Fox, An MCMC method for uncertainty quantification in
    nonnegativity constrained inverse problems, Inverse Probl. Sci. Eng. 20 (2012)
    """
    Sigma = np.linalg.inv(invSigma)

    try:
        #'lower' syntax is faster for sparse matrices. Also matches convention in
        # Bardsley paper.
        C_L = np.linalg.cholesky(Sigma)
    except:
        C_L = sqrtm(Sigma)
        
    v = np.random.standard_normal(size=(len(tauKtX),))
    w = np.linalg.solve(np.matrix.transpose(C_L),v)
   
    P = fnnls(invSigma,tauKtX+w)

    return P

def fnnls(AtA,Atb,tol=[],maxiter=[],verbose=False):
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