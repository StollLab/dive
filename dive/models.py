from re import L
import pymc3 as pm
import numpy as np
import deerlab as dl

from .utils import *
from .deer import *

def model(t, Vexp, pars,default_r=False):
    """
    Returns a dictionary m that contains the DEER data in m['t'] and m['Vexp']
    and the PyMC3 model in m['model']. Additional (constant) model parameters
    are in m['pars'].
    The data are rescaled internally to max(Vexp)==1.
    """
    
    # Rescale data to max 1
    Vscale = np.amax(Vexp)
    Vexp /= Vscale

    if "method" not in pars:
        raise KeyError("'method' is a required field.")
    method = pars["method"]

    if "r" not in pars:
        raise KeyError(f"r is a required key for ""method"" = ""{method}"".")

    if method == "gaussian":
        if "nGauss" not in pars:
           raise KeyError(f"nGauss is a required key for ""method"" = ""{method}"".") 
        nGauss = pars["nGauss"]

        r = pars["r"]
        K0 = dl.dipolarkernel(t,r,integralop=True)
        model_pymc = multigaussmodel(t, Vexp, K0, r, nGauss)
        
        model_pars = {"K0": K0, "r": r, "ngaussians": nGauss}

    elif method == "regularization" or method == "regularization2":
        if default_r:
            r = pars["r"]
            dt= max(t)-min(t)
            rmax= (108*dt)**0.333333333333333
            print(f"rmax: {rmax}")
            rmin = min(r)
            num = len(r)
            dr = (max(r)-rmin)/num
            print(f"dr:  {dr}")
            num = int((rmax-rmin)/dr)
            print(f"num: {num}")

            r= np.linspace(rmin,rmax,num)

        else:
            r = pars["r"]
        
        K0 = dl.dipolarkernel(t, r,integralop=False)
        L = dl.regoperator(np.arange(len(r)), 2, includeedges=False)
        LtL = L.T@L
        K0tK0 = K0.T@K0

        delta_prior = [1, 1e-6]
        tau_prior = [1, 1e-4]
        
        tauGibbs = method == "regularization"
        deltaGibbs = method == "regularization"
        model_pymc = regularizationmodel(t, Vexp, K0, r, delta_prior=delta_prior, tau_prior=tau_prior, tauGibbs=tauGibbs, deltaGibbs=deltaGibbs)

        model_pars = {"r": r, "K0": K0, "L": L, "LtL": LtL, "K0tK0": K0tK0, "delta_prior": delta_prior, "tau_prior": tau_prior}
    
    else:
        raise ValueError(f"Unknown method '{method}'.")
    
    model_pars['method'] = method
    model_pars['Vscale'] = Vscale

    model = {'model': model_pymc, 'pars': model_pars, 't': t, 'Vexp': Vexp}
    
    print(f"Time range:         {len(t):4d} points from {min(t):g} µs to {max(t):g} µs")
    print(f"Distance range:     {len(r):4d} points from {min(r):g} nm to {max(r):g} nm")
    print(f"Model:              {method}")
    if method == "gaussian":
        print(f"Number of Gaussian: {nGauss}")
    
    return model

def multigaussmodel(t, Vdata, K0, r, nGauss=1,
        includeBackground=True, includeModDepth=True, includeAmplitude=True,
    ):
    """
    Generates a PyMC3 model for a DEER signal over time vector t
    (in µs) given data in Vdata.
    It uses a multi-Gaussian distributions, where nGauss is the number
    of Gaussians, plus an exponential background.
    """
    
    # Parameters for r0 prior
    r0min = 1.3
    r0max = 7

    # Model definition
    with pm.Model() as model:
        
        # Distribution parameters
        r0_rel = pm.Beta('r0_rel', alpha=2, beta=2, shape=nGauss)
        r0 = pm.Deterministic('r0', r0_rel.sort()*(r0max-r0min) + r0min)        
        w = pm.Bound(pm.InverseGamma, lower=0.05, upper=3.0)('w', alpha=0.1, beta=0.2, shape=nGauss)

        if nGauss>1:
            a = pm.Dirichlet('a', a=np.ones(nGauss))
        
        # Calculate distance distribution
        if nGauss==1:
            P = gauss(r,r0,FWHM2sigma(w))
        else:
            P = np.zeros(np.size(K0,1))
            for i in range(nGauss):
                P += a[i]*gauss(r,r0[i],FWHM2sigma(w[i]))
        pm.Deterministic('P', P)  # for reporting
        
        # Time-domain model signal
        Vmodel = pm.math.dot(K0,P)

        # Add modulation depth
        if includeModDepth:
            lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
            Vmodel = (1-lamb) + lamb*Vmodel

        # Add background
        if includeBackground:
            k = pm.Gamma('k', alpha=0.5, beta=2)
            B = bg_exp(t,k)
            Vmodel *= B
        
        # Add overall amplitude
        if includeAmplitude:
            V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1, sigma=0.2)
            Vmodel *= V0
        
        # Noise level
        sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
        
        # Likelihood
        pm.Normal('V', mu=Vmodel, sigma=sigma, observed=Vdata)
        
    return model

def regularizationmodel(t, Vdata, K0, r,
        delta_prior=None, tau_prior=None,
        includeBackground=True, includeModDepth=True, includeAmplitude=True,
        tauGibbs=True, deltaGibbs=True,
    ):
    """
    Generates a PyMC3 model for a DEER signal over time vector t (in µs) given data in Vdata.
    Model parameters:
      P      distance distribution vector (nm^-1)
      tau    noise precision (inverse of noise variance)
      delta  smoothing hyperparameter (= alpha^2/sigma^2)
      lamb   modulation amplitude
      k      background decay rate (µs^-1)
      V0     overall amplitude
    """
    
    dr = r[1]-r[0]
    
    # Model definition
    with pm.Model() as model:
        # Distance distribution
        testval  = np.zeros(len(r))
        P = pm.NoDistribution('P', shape=len(r), dtype='float64',testval=testval) # no prior (it's included in the Gibbs sampler)
        
        # Time-domain model signal
        Vmodel = pm.math.dot(K0*dr,P)

        # Add modulation depth
        if includeModDepth:
            lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
            Vmodel = (1-lamb) + lamb*Vmodel
        
        # Add background
        if includeBackground:
            k = pm.Gamma('k', alpha=0.5, beta=2)            
            B = bg_exp(t, k)
            Vmodel *= B
            
        # Add overall amplitude
        if includeAmplitude:
            V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1, sigma=0.2)
            Vmodel *= V0
            
        # Noise parameter
        if tauGibbs:
            tau = pm.NoDistribution('tau', shape=(), dtype='float64', testval=1.0) # no prior (it's included in the Gibbs sampler)
        else:
            tau = pm.Gamma('tau', alpha=tau_prior[0], beta=tau_prior[1])
        sigma = pm.Deterministic('sigma', 1/np.sqrt(tau)) # for reporting

        # Regularization parameter
        if deltaGibbs:
            delta = pm.NoDistribution('delta', shape=(), dtype='float64', testval=1.0)
        else:
            delta = pm.Gamma('delta', alpha=delta_prior[0], beta=delta_prior[1])
        lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)) )  # for reporting
        
        # Add likelihood
        likelihood = pm.Normal('V', mu=Vmodel, tau=tau, observed=Vdata)
        
    return model
