import pymc as pm

import numpy as np
import deerlab as dl

from .utils import *
from .deer import *
from .samplers import *

def model(t, Vexp, pars):
    """
    Returns a dictionary m that contains the DEER data in m['t'] and m['Vexp']
    and the PyMC model in m['model']. Additional (constant) model parameters
    are in m['pars'].
    The data are rescaled internally to max(Vexp)==1.
    """
    
    # Rescale data to max 1
    Vscale = np.amax(Vexp)
    Vexp_scaled = Vexp/Vscale

    if "method" not in pars:
        raise KeyError("'method' is a required field.")
    method = pars["method"]

    if "r" not in pars:
        raise KeyError(f"r is a required key for ""method"" = ""{method}"".")
    
    # Supplement defaults
    rmax_opt = pars["rmax_opt"] if "rmax_opt" in pars else "user"
    bkgd_var = pars["bkgd_var"] if "bkgd_var" in pars else "Bend"

    if rmax_opt == "auto":
        r_ = pars["r"]
        tmax = max(abs(t))
        rmax= (108*tmax)**0.333333333333333
        dr = (max(r_)-min(r_))/len(r_)
        num = int((rmax-min(r_))/dr)
        r = np.linspace(min(r_),rmax,num)

    elif rmax_opt == "user":
        r = pars["r"]

    else:
        raise ValueError(f"Unknown rmax selection method '{rmax_opt}'.")

    if method == "gaussian":
        if "nGauss" not in pars:
           raise KeyError(f"nGauss is a required key for ""method"" = ""{method}"".") 
        nGauss = pars["nGauss"]

        K0 = dl.dipolarkernel(t, r, integralop=True)
        model_pymc = multigaussmodel(t, Vexp_scaled, K0, r, nGauss)
        
        model_pars = {"K0": K0, "r": r, "ngaussians": nGauss}

    elif method == "regularization" or method == "regularizationP":

        K0 = dl.dipolarkernel(t, r,integralop=False)
        L = dl.regoperator(np.arange(len(r)), 2, includeedges=False)
        LtL = L.T@L
        K0tK0 = K0.T@K0

        delta_prior = [1, 1e-6]
        tau_prior = [1, 1e-4]
        
        tauGibbs = method == "regularization"
        deltaGibbs = method == "regularization"
        model_pymc = regularizationmodel(t, Vexp_scaled, K0, r, delta_prior=delta_prior, tau_prior=tau_prior, tauGibbs=tauGibbs, deltaGibbs=deltaGibbs)

        model_pars = {"r": r, "K0": K0, "L": L, "LtL": LtL, "K0tK0": K0tK0, "delta_prior": delta_prior, "tau_prior": tau_prior}
    
    else:
        raise ValueError(f"Unknown method '{method}'.")
    
    model_pars['method'] = method
    model_pars['Vscale'] = Vscale
    model_pars['Vexp'] = Vexp_scaled
    model_pars['t'] = t
    model_pars['dr'] = r[1]-r[0]

    model = {'model': model_pymc, 'pars': model_pars, 't': t, 'Vexp': Vexp_scaled}
    
    # Print information about data and model
    print(f"Time range:         {min(t):g} µs to {max(t):g} µs  ({len(t):d} points, step size {t[1]-t[0]:g} µs)")
    print(f"Distance range:     {min(r):g} nm to {max(r):g} nm  ({len(r):d} points, step size {r[1]-r[0]:g} nm)")
    print(f"Vexp max:           {Vscale:g}")
    print(f"Background:         exponential")
    if method == "gaussian":
        print(f"P model:            {nGauss} Gaussians")
    else:
        print(f"P model:            {method}")
    
    return model

def multigaussmodel(t, Vdata, K0, r, nGauss=1,
        includeBackground=True, includeModDepth=True, includeAmplitude=True,
    ):
    """
    Generates a PyMC model for a DEER signal over time vector t
    (in µs) given data in Vdata.
    It uses a multi-Gaussian distributions, where nGauss is the number
    of Gaussians, plus an exponential background.
    """

    # Model definition
    with pm.Model() as model:
        
        # Distance distribution parameters
        r0_rel = pm.Beta('r0_rel', alpha=2, beta=2, shape=nGauss)
        r0 = pm.Deterministic('r0', r0_rel.sort()*(max(r)-min(r)) + min(r))  # for reporting
        w = pm.TruncatedNormal('w', pm.InverseGamma('w_mu', alpha=0.1, beta=0.2, shape=nGauss), lower=0.05, upper=3.0, shape=nGauss)
        #w = pm.Truncated('w', pm.InverseGamma.dist(alpha=0.1, beta=0.5, shape=nGauss), lower=0.02, upper=4.0)   # Old definition

        if nGauss>1:
            a = pm.Dirichlet('a', a=np.ones(nGauss))
        
        # Calculate distance distribution
        if nGauss==1:
            P = gauss(r, r0, FWHM2sigma(w))
        else:
            P = np.zeros(np.size(K0,1))
            for i in range(nGauss):
                P += a[i]*gauss(r, r0[i], FWHM2sigma(w[i]))
        pm.Deterministic('P', P)  # for reporting
        
        # Time-domain model signal
        Vmodel = pm.math.dot(K0,P)

        # Add modulation depth
        if includeModDepth:
            lamb = pm.Beta('lamb', alpha=1.3, beta=2.0, initval=0.2)
            Vmodel = (1-lamb) + lamb*Vmodel

        # Add background
        if includeBackground:
            Bend = pm.Beta("Bend", alpha=1.0, beta=1.5)
            k = pm.Deterministic('k', -np.log(Bend)/t[-1])  # for reporting
            B = bg_exp(t,k)
            Vmodel *= B

        # Add overall amplitude
        if includeAmplitude:
            V0 = pm.TruncatedNormal('V0', mu=1, sigma=0.2, lower=0)
            Vmodel *= V0
        
        # Noise level
        #sigma = pm.Gamma('sigma', alpha=1, beta=0.1)  # old prior
        sigma = pm.Gamma('sigma', alpha=0.7, beta=2)

        # Likelihood
        pm.Normal('V', mu=Vmodel, sigma=sigma, observed=Vdata)
        
    return model

def regularizationmodel(t, Vdata, K0, r,
        delta_prior=None, tau_prior=None,
        includeBackground=True, includeModDepth=True, includeAmplitude=True,
        tauGibbs=True, deltaGibbs=True, bkgd_var="Bend"
    ):
    """
    Generates a PyMC model for a DEER signal over time vector t (in µs) given data in Vdata.
    Model parameters:
      P      distance distribution vector (nm^-1)
      tau    noise precision (inverse of noise variance)
      delta  smoothing hyperparameter (= alpha^2/sigma^2)
      lamb   modulation amplitude
      k      background decay rate constant (µs^-1)
      Bend   background decay value at end of time interval
      V0     overall amplitude
    """
    
    dr = r[1]-r[0]
    
    # Model definition
    with pm.Model() as model:
        # Distance distribution - no prior (it's included in the Gibbs sampler)
        P = pm.MvNormal('P', shape=len(r), mu=np.zeros(len(r)), cov=np.identity(len(r)))
        
        # Time-domain model signal
        Vmodel = pm.math.dot(K0*dr,P)

        # Add modulation depth
        if includeModDepth:
            lamb = pm.Beta('lamb', alpha=1.3, beta=2.0, initval=0.2)
            Vmodel = (1-lamb) + lamb*Vmodel

        # Add background
        if includeBackground:
            Bend = pm.Beta("Bend", alpha=1.0, beta=1.5)
            k = pm.Deterministic('k', -np.log(Bend)/t[-1])
            B = bg_exp(t,k)
            Vmodel *= B

        # Add overall amplitude
        if includeAmplitude:
            V0 = pm.TruncatedNormal('V0', mu=1, sigma=0.2, lower=0)
            Vmodel *= V0
            
        # Noise parameter
        if tauGibbs: # no prior (it's included in the Gibbs sampler)
            tau = pm.Flat('tau', initval=1.2)
        else:
            tau = pm.Gamma('tau', alpha=tau_prior[0], beta=tau_prior[1], initval=1.3)
        sigma = pm.Deterministic('sigma', 1/np.sqrt(tau))  # for reporting

        # Regularization parameter
        if deltaGibbs: # no prior (it's included in the Gibbs sampler)
            delta = pm.Flat('delta', initval=1.02)
        else:
            delta = pm.Gamma('delta', alpha=delta_prior[0], beta=delta_prior[1], initval=1.02)
        lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)) )  # for reporting
        
        # Add likelihood
        pm.Normal('V', mu=Vmodel, tau=tau, observed=Vdata)
        
    return model


def sample(model_dic, MCMCparameters, steporder=None, NUTSpars=None, seed=None):
    """
    Use PyMC to draw samples from the posterior for the model, according to the parameters provided with MCMCparameters.
    """
    
    # Complain about missing required keywords
    requiredKeys = ["draws", "tune", "chains"]
    for key in requiredKeys:
        if key not in MCMCparameters:
            raise KeyError(f"The required MCMC parameter '{key}' is missing.")
    
    # Supplement defaults for optional keywords
    defaults = {"cores": 2, "progressbar": True}
    MCMCparameters = {**defaults, **MCMCparameters}
    
    model = model_dic['model']
    model_pars = model_dic['pars']
    method = model_pars['method']
    
    # Set stepping methods, depending on model
    if method == "gaussian":
        
        removeVars  = ["r0_rel", "w_mu"]
        
        with model:
            NUTS_varlist = [model['r0_rel'], model['w'], model['w_mu']]
            if model_pars['ngaussians']>1:
                NUTS_varlist.append(model['a'])
            NUTS_varlist.append(model['sigma'])
            NUTS_varlist.append(model['Bend'])
            NUTS_varlist.append(model['V0'])
            NUTS_varlist.append(model['lamb'])
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
    
        step = [step_NUTS]
        
    elif method == "regularization":
        
        removeVars = None
        
        with model:
            
            conjstep_tau = randTau_posterior(model_pars)
            conjstep_P = randPnorm_posterior(model_pars)
            conjstep_delta = randDelta_posterior(model_pars)
            
            NUTS_varlist = [model['V0'], model['lamb'], model['Bend']]
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist, on_unused_input="ignore")
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, on_unused_input="ignore", **NUTSpars)
            
        step = [step_NUTS, conjstep_P, conjstep_tau, conjstep_delta]
        if steporder is not None:
            step = [step[i] for i in steporder]
        
    elif method == "regularizationP":
        
        removeVars = None
        
        with model:
                
            conjstep_P = randPnorm_posterior(model_pars)
            
            NUTS_varlist = [model['tau'], model['delta'], model['Bend'], model['V0'], model['lamb']]
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
                
        step = [conjstep_P, step_NUTS]
        if steporder is not None:
            step = [step[i] for i in steporder]
            
    else:
        
        raise KeyError(f"Unknown method '{method}'.",method)

    # Perform MCMC sampling
    idata = pm.sample(model=model, step=step, random_seed=seed, **MCMCparameters)

    # Remove undesired variables
    if removeVars is not None:
        for key in removeVars:
            if key in idata.posterior:
                del idata.posterior[key]

    return idata
