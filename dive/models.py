import pymc3 as pm
import numpy as np
import math as m
import deerlab as dl

from .utils import *
from .deer import *

import theano.tensor as tt

def model(t, Vdata, pars):
    
    # Rescale data to max 1
    Vscale = np.amax(Vdata)
    Vdata /= Vscale

    if 'method' not in pars:
        raise KeyError("'method' is a required field.")
    method = pars['method']

    if method == 'gaussian':
        if 'nGauss' not in pars:
           raise KeyError(f"nGauss is a required key for 'method' = '{method}'.") 

        r = np.linspace(1,10,451)
        K0 = dl.dipolarkernel(t,r)
        model_graph = multigaussmodel(pars['nGauss'], t, Vdata, K0, r)
        
        model_pars = {'K0': K0, "r": r, "ngaussians": pars['nGauss']}

    elif method == 'regularization':
        if 'r' not in pars:
           raise KeyError(f"r is a required key for 'method' = '{method}'.")

        K0 = dl.dipolarkernel(t, pars['r'], integralop=False)
        model_graph = regularizationmodel(t, Vdata, K0, pars['r'])

        a_delta = 1
        b_delta = 1e-6
        a_tau = 1
        b_tau = 1e-4
        L = dl.regoperator(np.linspace(1,len(pars['r']),len(pars['r'])), 2)
        LtL = L.T@L
        K0tK0 = K0.T@K0

        model_pars = {'K0': K0, 'L': L, 'LtL': LtL, 'K0tK0': K0tK0, "r": pars['r'], 'a_delta': a_delta, 'b_delta': b_delta, 'a_tau': a_tau, 'b_tau': b_tau}
    
    else:
        raise ValueError(f"Unknown method '{method}'.")
    
    model_pars['method'] = method
    model_pars['Vscale'] = Vscale

    model = {'model_graph': model_graph, 'model_pars': model_pars, 't': t, 'Vexp': Vdata}
    return model

def multigaussmodel(nGauss, t, Vdata, K0, r):
    """
    Generates a PyMC3 model for a DEER signal over time vector t
    (in microseconds) given data in Vdata.
    It uses a multi-Gaussian distributions, where nGauss is the number
    of Gaussians, plus an exponential background.
    """
    r0min = 1.3
    r0max = 7    

    # Model definition
    model = pm.Model()
    with model:
        
        # Distribution model
        r0_rel = pm.Beta('r0_rel', alpha=2, beta=2, shape=nGauss)
        r0 = pm.Deterministic('r0', r0_rel.sort()*(r0max-r0min) + r0min)
        
        w = pm.Bound(pm.InverseGamma, lower=0.05, upper=3.0)('w', alpha=0.1, beta=0.2, shape=nGauss) # this is the FWHM of the Gaussian
        
        if nGauss>1:
            a = pm.Dirichlet('a', a=np.ones(nGauss))
        else:
            a = np.ones(1)
        
        # Calculate distance distribution
        if nGauss==1:
            P = gauss(r,r0,FWHM2sigma(w))
        else:
            P = np.zeros(np.size(K0,1))
            for i in range(nGauss):
                P += a[i]*gauss(r,r0[i],FWHM2sigma(w[i]))
        
        # Background model
        k = pm.Gamma('k', alpha=0.5, beta=2)
        B = bg_exp(t,k)
        
        # DEER signal
        lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
        V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1, sigma=0.2)
        
        Vmodel = deerTrace(pm.math.dot(K0,P),B,V0,lamb)

        sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
        
        # Likelihood
        pm.Normal('V', mu=Vmodel, sigma=sigma, observed=Vdata)
        
    return model

def regularizationmodel(t, Vdata, K0, r):
    """
    Generates a PyMC3 model for a DEER signal over time vector t
    (in microseconds) given data in Vdata.
    Model parameters:
      P      distance distribution vector (nm^-1)
      tau    noise precision (inverse of noise variance)
      delta  smoothing hyperparameter (= alpha^2/sigma^2)
      lamb   modulation amplitude
      k      background decay rate (µs^-1)
      V0     overall amplitude
    This model is intended to be used with Gibbs sampling with
    separate independent sampling steps for P, delta, and tau,
    plus a NUTS step for (k, lambda, V0).
    """

    dr = r[1] - r[0]
    
    # Model definition
    with pm.Model() as model:
        # Noise parameter -----------------------------------------------------
        tau = pm.NoDistribution('tau', shape=(), dtype='float64', testval=1.0) # no prior (it's included in the custom sampler)
        sigma = pm.Deterministic('sigma',1/np.sqrt(tau)) # for reporting only

        # Regularization parameter --------------------------------------------
        delta = pm.NoDistribution('delta', shape=(), dtype='float64', testval=1.0) # no prior (it's included in the custom sampler)
        lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)) )  # for reporting only
        
        # Time-domain parameters ----------------------------------------------
        lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
        V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1, sigma=0.2)

        # Background parameters -----------------------------------------------
        k = pm.Gamma('k', alpha=0.5, beta=2)

        # Distance distribution -----------------------------------------------
        P = pm.NoDistribution('P', shape=len(r), dtype='float64', testval=np.zeros(len(r))) # no prior (it's included in the custom sampler)

        # Calculate kernel matrix ---------------------------------------------
        B = dl.bg_exp(t, k)
        B_ = tt.tile(B,(len(r),1)).T
        Kintra = (1-lamb) + lamb*K0
        K = V0*Kintra*B_*dr

        # Time-domain signal --------------------------------------------------
        Vmodel = pm.math.dot(K,P)

        # Likelihood ----------------------------------------------------------
        pm.Normal('V', mu=Vmodel, tau=tau, observed=Vdata)
        
    return model
