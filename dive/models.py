from re import L
import pymc3 as pm
import numpy as np
import deerlab as dl

from .utils import *
from .deer import *

def model(t, Vexp, pars):
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
    
    # Supplement defaults
    rmax_opt = pars["rmax_opt"] if "rmax_opt" in pars else "user"
    bkgd_var = pars["bkgd_var"] if "bkgd_var" in pars else "Bend"

    if rmax_opt == "auto":
        r_ = pars["r"]
        tmax = max(t)-min(t)
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

        K0 = dl.dipolarkernel(t,r,integralop=True)
        model_pymc = multigaussmodel(t, Vexp, K0, r, nGauss, bkgd_var)
        
        model_pars = {"K0": K0, "r": r, "ngaussians": nGauss, "bkgd_var": bkgd_var}

    elif method == "regularization" or method == "regularization2":

        K0 = dl.dipolarkernel(t, r,integralop=False)
        L = dl.regoperator(np.arange(len(r)), 2, includeedges=False)
        LtL = L.T@L
        K0tK0 = K0.T@K0

        delta_prior = [1, 1e-6]
        tau_prior = [1, 1e-4]
        
        tauGibbs = method == "regularization"
        deltaGibbs = method == "regularization"
        model_pymc = regularizationmodel(t, Vexp, K0, r, delta_prior=delta_prior, tau_prior=tau_prior, tauGibbs=tauGibbs, deltaGibbs=deltaGibbs, bkgd_var=bkgd_var)

        model_pars = {"r": r, "K0": K0, "L": L, "LtL": LtL, "K0tK0": K0tK0, "delta_prior": delta_prior, "tau_prior": tau_prior, "bkgd_var": bkgd_var}
    
    else:
        raise ValueError(f"Unknown method '{method}'.")
    
    model_pars['method'] = method
    model_pars['Vscale'] = Vscale

    model = {'model': model_pymc, 'pars': model_pars, 't': t, 'Vexp': Vexp}
    
    print(f"Time range:         {len(t):4d} points (dt={(max(t)-min(t))/len(t):g}) from {min(t):g} µs to {max(t):g} µs")
    print(f"Distance range:     {len(r):4d} points (dr={(max(r)-min(r))/len(r):g}) from {min(r):g} nm to {max(r):g} nm")
    print(f"Model:              {method}")
    if method == "gaussian":
        print(f"Number of Gaussian: {nGauss}")
    
    return model

def multigaussmodel(t, Vdata, K0, r, nGauss=1, bkgd_var="k",
        includeBackground=True, includeModDepth=True, includeAmplitude=True,
    ):
    """
    Generates a PyMC3 model for a DEER signal over time vector t
    (in µs) given data in Vdata.
    It uses a multi-Gaussian distributions, where nGauss is the number
    of Gaussians, plus an exponential background.
    """

    # Model definition
    with pm.Model() as model:
        
        # Distribution parameters
        r0_rel = pm.Beta('r0_rel', alpha=2, beta=2, shape=nGauss)
        r0 = pm.Deterministic('r0', r0_rel.sort()*(max(r)-min(r)) + min(r))        
        w = pm.Bound(pm.InverseGamma, lower=0.05, upper=3.0)('w', alpha=0.1, beta=0.2, shape=nGauss)

        # Old multigauss model definition for w
        #w = pm.Bound(pm.InverseGamma,lower=0.02,upper=4.0)('w',alpha=0.1,beta=0.5,shape=nGauss)
        #BoundedInvGamma = pm.Bound(pm.InverseGamma, lower=0.02, upper=4.0)
        #w = BoundedInvGamma('w', alpha=0.1, beta=0.5,shape=nGauss)
        

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

            if bkgd_var == "k":
                # Old k prior
                #k = pm.Gamma('k',alpha=1,beta=0.05)

                k = pm.Gamma('k', alpha=0.5, beta=2)
                B = bg_exp(t,k)
                Vmodel *= B

            elif bkgd_var == "tauB":
                tauB = pm.Gamma('tauB', alpha=0.5, beta=0.01)
                B = bg_exp_time(t,tauB)
                Vmodel *= B

            elif bkgd_var == "Bend":
                #Bend = pm.Uniform('Bend',lower=0.0,upper=1.0)
                Bend = pm.Beta("Bend",alpha=1.0,beta=1.5)
                #k = pm.Deterministic('k',(1/np.max(t))*np.log((1-lamb)/Bend))
                k = pm.Deterministic('k',-1/t[-1]*np.log(Bend))
                B = bg_exp(t,k)
                Vmodel *= B
                
            else:
                raise ValueError(f"Unknown background method '{bkgd_var}'.")

        # Add overall amplitude
        if includeAmplitude:
            V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1, sigma=0.2)
            Vmodel *= V0
        
        # Noise level
        sigma = pm.Gamma('sigma', alpha=0.7, beta=2)

        # Old prior
        #sigma = pm.Gamma('sigma', alpha=1, beta=0.1)

        # Likelihood
        pm.Normal('V', mu=Vmodel, sigma=sigma, observed=Vdata)
        
    return model

def regularizationmodel(t, Vdata, K0, r,
        delta_prior=None, tau_prior=None,
        includeBackground=True, includeModDepth=True, includeAmplitude=True,
        tauGibbs=True, deltaGibbs=True, bkgd_var="Bend"
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
            #lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
            lamb = pm.Beta('lamb',alpha=1.0,beta=2.0)
            Vmodel = (1-lamb) + lamb*Vmodel

        # Add background
        if includeBackground:
            
            if bkgd_var == "k":
                k = pm.Gamma('k', alpha=0.5, beta=2)            
                B = bg_exp(t, k)
                Vmodel *= B

            elif bkgd_var == "tauB":  
                #tauB = pm.Gamma('tauB', alpha=0.7, beta=0.05)
                #tauB = pm.Gamma('tauB', alpha=0.7, beta=0.1)

                tauB = pm.Gamma('tauB',alpha=0.9,beta=0.02)
                B = bg_exp_time(t, tauB)
                Vmodel *= B

            elif bkgd_var == "Bend":
                #Bend = pm.Uniform('Bend',lower=0.0,upper=1.0)
                Bend = pm.Beta("Bend",alpha=1.0,beta=1.5)
                #k = pm.Deterministic('k',(1/np.max(t))*np.log((1-lamb)/Bend))
                k = pm.Deterministic('k',-1/t[-1]*np.log(Bend))
                B = bg_exp(t,k)
                Vmodel *= B
            
            else: 
                raise ValueError(f"Unknown background method '{bkgd_var}'.")

        # Add overall amplitude
        if includeAmplitude:
            V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1.0, sigma=0.2)
            Vmodel *= V0
            
        # Noise parameter
        if tauGibbs:
            tau = pm.NoDistribution('tau', shape=(), dtype='float64', testval=1.0) # no prior (it's included in the Gibbs sampler)
            #tau = pm.Gamma('tau', alpha=tau_prior[0], beta=tau_prior[1])
        else:
            tau = pm.Gamma('tau', alpha=tau_prior[0], beta=tau_prior[1])
        sigma = pm.Deterministic('sigma', 1/np.sqrt(tau)) # for reporting

        # Regularization parameter
        if deltaGibbs:
            delta = pm.NoDistribution('delta', shape=(), dtype='float64', testval=1.0)
            #delta = pm.Gamma('delta', alpha=delta_prior[0], beta=delta_prior[1]) # no prior (it's included in the Gibbs sampler)
        else:
            delta = pm.Gamma('delta', alpha=delta_prior[0], beta=delta_prior[1])
        lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)) )  # for reporting
        
        # Add likelihood
        likelihood = pm.Normal('V', mu=Vmodel, tau=tau, observed=Vdata)
        
    return model
