import pymc3 as pm
import numpy as np
import deerlab as dl

from .utils import *
from .deer import *

def model(t, Vdata, pars):
    
    # Rescale data to max 1
    Vscale = np.amax(Vdata)
    Vdata /= Vscale

    if "method" not in pars:
        raise KeyError("'method' is a required field.")
    method = pars["method"]

    if method == "gaussian":
        if "nGauss" not in pars:
           raise KeyError(f"nGauss is a required key for ""method"" = ""{method}"".") 
        if "r" not in pars:
           raise KeyError(f"r is a required key for ""method"" = ""{method}"".")

        r = pars["r"]
        K0 = dl.dipolarkernel(t,r,integralop=True)
        model_pymc = multigaussmodel(t, Vdata, K0, r, pars["nGauss"])
        
        model_pars = {"K0": K0, "r": r, "ngaussians": pars["nGauss"]}

    elif method == "regularization" or method == "regularization2":
        if "r" not in pars:
           raise KeyError(f"r is a required key for ""method"" = ""{method}"".")

        #r = np.linspace(1,10,451)
        r = pars["r"]
        K0 = dl.dipolarkernel(t, r,integralop=False)
        L = dl.regoperator(np.arange(len(r)), 2)
        LtL = L.T@L
        K0tK0 = K0.T@K0

        delta_prior = [1, 1e-6]
        tau_prior = [1, 1e-4]
        
        tauGibbs = method == "regularization"
        deltaGibbs = method == "regularization"
        model_pymc = regularizationmodel(t, Vdata, K0, r, delta_prior=delta_prior, tau_prior=tau_prior, tauGibbs=tauGibbs, deltaGibbs=deltaGibbs)

        model_pars = {"r": r, "K0": K0, "L": L, "LtL": LtL, "K0tK0": K0tK0, "delta_prior": delta_prior, "tau_prior": tau_prior}
    
    else:
        raise ValueError(f"Unknown method '{method}'.")
    
    model_pars['method'] = method
    model_pars['Vscale'] = Vscale

    model = {'model': model_pymc, 'pars': model_pars, 't': t, 'Vexp': Vdata}
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
        w = pm.Bound(pm.InverseGamma, lower=0.05, upper=3.0)('w', alpha=0.1, beta=0.2, shape=nGauss) # this is the FWHM of the Gaussian
        if nGauss>1:
            a = pm.Dirichlet('a', a=np.ones(nGauss))
        
        # Calculate distance distribution
        if nGauss==1:
            P = gauss(r,r0,FWHM2sigma(w))
        else:
            P = np.zeros(np.size(K0,1))
            for i in range(nGauss):
                P += a[i]*gauss(r,r0[i],FWHM2sigma(w[i]))
        
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
        P = pm.NoDistribution('P', shape=len(r), dtype='float64', testval=np.zeros(len(r))) # no prior (it's included in the Gibbs sampler)

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
        sigma = pm.Deterministic('sigma',1/np.sqrt(tau)) # for reporting

        # Regularization parameter
        if deltaGibbs:
            delta = pm.NoDistribution('delta', shape=(), dtype='float64', testval=1.0) # no prior (it's included in the Gibbs sampler)
        else:
            delta = pm.Gamma('delta', alpha=delta_prior[0], beta=delta_prior[1])
        lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)) )  # for reporting
        
        # Add likelihood
        pm.Normal('V', mu=Vmodel, tau=tau, observed=Vdata)
        
    return model
