from dataclasses import replace
import numpy as np
import math as m
from scipy.special import fresnel
from datetime import date
import os

from .constants import *
from .deerload import *

import arviz as az
from .plotting import *

def addnoise(V,sig):

    """
    Add Gaussian noise with standard deviation sig to signal
    """
    noise = np.random.normal(0, sig, np.size(V))
    Vnoisy = V + noise
    return Vnoisy


def FWHM2sigma(FWHM):
    """
    Convert the full width at half maximum, FWHM, of a Gaussian to the standard deviation, sigma.
    """

    sigma = FWHM/(2*m.sqrt(2*m.log(2)))

    return sigma


def sigma2FWHM(sigma):
    """
    Convert the standard deviation, sigma, of a Gaussian to the full width at half maximum, FWHM.
    """
    FWHM = sigma/(2*m.sqrt(2*m.log(2)))

    return FWHM


def dipolarkernel(t,r):
    """
    K = dipolarkernel(t,r)
    Calculate dipolar kernel matrix.
    Assumes t in microseconds and r in nanometers
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


def loadTrace(FileName):
    """
    Load a DEER trace, can be a Bruker or comma separated file.
    """
    if FileName.endswith('.dat') or FileName.endswith('.txt') or FileName.endswith('.csv'):
        data = np.genfromtxt(FileName, delimiter=',', skip_header=1)
        t = data[:,0]
        Vdata = data[:,1]
    elif FileName.endswith('.DTA') or FileName.endswith('.DSC'):
        t, Vdata, Parameters = deerload(FileName)
    else:
        raise ValueError('The file format is not recognized.')

    return t, Vdata


def interpret(trace,model_dic):
    
    class FitResult:
        def __init__(self,trace, model):
            d = {key: trace.posterior[key] for key in trace.posterior}
            self.__dict__.update(d)

            self.r = model['pars']['r']
            self.t = model['t']
            self.Vexp = model['Vexp']
            self.varnames = trace.posterior
            self.trace = trace
            self.K = dl.dipolarkernel(self.t, self.r)
            self.dr = self.r[1] - self.r[0]

            # self.plots = Plots(trace,model)

        def subsample_fits(self, n=100, seed=1):
            np.random.seed(seed)
            idxs = np.random.choice(len(self.trace), n, replace=False)
            Ps = [self.P[idx].copy() for idx in idxs]
            Bs, Vs = [], []


            for idx in idxs:
                V_ = self.K@self.P[idx]

                if 'lamb' in self.varnames:
                    V_ = (1-self.lamb[idx]) + self.lamb[idx]*V_

                if 'k' in self.varnames:
                    B = dl.bg_exp(self.t, self.k[idx])
                    V_ *= B
                    
                    Blamb = (1-self.lamb[idx])*B
                
                if 'V0' in self.varnames:
                    Blamb *= self.V0[idx]
                    Bs.append(Blamb)
                    V_ *= self.V0[idx]
                Vs.append(V_)

            return Vs, Bs, Ps

        def plot(self,style = 'noodle',j =0.95):
            plt.style.use('seaborn-darkgrid')
            
            plt.rcParams.update({'font.family':'serif'})
            if style == 'noodle':
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))
                Vs, Bs, Ps = self.subsample_fits()
                
                ax1.plot(self.t, self.Vexp,'g.',linewidth=0.5,alpha = 0.3)

                for V, B, P in zip(Vs, Bs, Ps):
                    ax2.plot(self.r, P, 'cornflowerblue', linewidth=0.3)
                    ax1.plot(self.t, V,'#0000EE',linewidth=0.3)
                    ax1.plot(self.t, B,'#FAD02C',linewidth=0.3)
                    ax1.plot(self.t, V-self.Vexp,'#FF0080',linewidth=0.3)

                leg1= ax1.legend(['Data','Vexp','Background','Residuals'])
                leg2 = ax2.legend(['Distance Distribution'])

                for lh1,lh2 in zip(leg1.legendHandles,leg2.legendHandles): 
                    lh1.set_alpha(1)
                    lh2.set_alpha(1)
                

                #ax2.set_xlabel(r'time ($\rm\mus$)')

            if style == 'mean-ci':
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))
                Vs, Bs, Ps = self.subsample_fits()
                
                l0, = ax1.plot(self.t, self.Vexp,'#808080',marker='.',linewidth=0.5,alpha = 0.3,label = 'Data',linestyle='None')

                Pmean = np.mean(Ps,0)
                Phd = az.hdi(np.array(Ps),j) 

                Vmean = np.mean(Vs,0)
                Vhd = az.hdi(np.array(Vs),j) 
                

                Bmean = np.mean(Bs,0)
                Bhd = az.hdi(np.array(Bs),j) 

            
                ax2.plot(self.r, Pmean, '#8E05D4', linewidth=1)
                ax2.fill_between(self.r,Phd[:,0],Phd[:,1],alpha = 0.7)


                l1,=ax1.plot(self.t, Vmean, '#964B00', linewidth=0.5,label='Vexp mean')
                l2,=ax1.plot(self.t, Bmean, 'b', linewidth=0.5 ,label='Background mean')
                l3,=ax1.plot(self.t, Vmean-self.Vexp,'#FF0080',linewidth=0.8,label = "Residuals")


                ax1.fill_between(self.t,Vhd[:,0],Vhd[:,1],color = 'C0',alpha =0.5)
                ax1.fill_between(self.t,Bhd[:,0],Bhd[:,1],color = '#FFF68F',alpha =0.1)
                ax1.legend(handles=[l0,l1,l2,l3])
                ax2.legend()
                
                
                #ax2.set_xlabel(r'time ($\rm\mus$)')
            
            ax2.set_xlabel('Distance(nm)')
            ax2.set_ylabel("Probability($1/nm$)")
            ax2.xaxis.set_major_locator(plt.MaxNLocator(16))

            ax1.set_ylabel('Signal (a.u.)')
            ax1.set_xlabel("Time(µs)")
            ax1.xaxis.set_major_locator(plt.MaxNLocator(15))
           


            fig.tight_layout()
            plt.style.use('seaborn-darkgrid')
            return fig

        def summary(self):
            printsummary(self.trace,self.model)
            

    fit = FitResult(trace,model_dic)

    return fit


def saveTrace(df, Parameters, SaveName='empty'):
    """
    Save a trace to a CSV file.
    """
    if SaveName == 'empty':
        today = date.today()
        datestring = today.strftime("%Y%m%d")
        SaveName = "./traces/{}_traces.dat".format(datestring)
    
    if not SaveName.endswith('.dat'):
        SaveName = SaveName+'.dat'

    shape = df.shape 
    cols = df.columns.tolist()

    os.makedirs(os.path.dirname(SaveName), exist_ok=True)

    f = open(SaveName, 'a+')
    f.write("# Traces from the MCMC simulations with PyMC\n")
    f.write("# The following {} parameters were investigated:\n".format(shape[1]))
    f.write("# {}\n".format(cols))
    f.write("# nParameters nChains nIterations\n")
    if Parameters['nGauss'] == 1:
        f.write("{},{},{},0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 2:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 3:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 4:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))

    df.to_csv (f, index=False, header=False)

    f.close()

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
