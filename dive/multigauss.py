import pymc3 as pm
import numpy as np
import math as m

from .utils import *

def multigaussmodel(nGauss,t,Vdata):
    """
    Generates a PyMC3 model for a DEER signal over time vector t
    (in microseconds) given data in Vdata.
    It uses a multi-Gaussian distributions, where nGauss is the number
    of Gaussians, plus an exponential background.
    """
    
    # Rescale data to max 1
    Vscale = np.amax(Vdata)
    Vdata /= Vscale
    
    # Calculate dipolar kernel for integration
    r = np.linspace(1,10,451)
    K = dipolarkernel(t,r)        
    
    # Model definition
    model = pm.Model()
    with model:
        
        # Distribution model
        r0min = 1.3
        r0max = 7
        
        r0_rel = pm.Beta('r0_rel', alpha=2, beta=2, shape=nGauss)
        r0 = pm.Deterministic('r0', r0_rel.sort()*(r0max-r0min) + r0min)
        
        w = pm.Bound(pm.InverseGamma, lower=0.05, upper=3.0)('w', alpha=0.1, beta=0.2, shape=nGauss) # this is the FWHM of the Gaussian
        # sig = pm.Deterministic('sig_r', w/(2*m.sqrt(2*m.log(2)))) # this is the widht of the Gaussian as it is used 
        
        if nGauss>1:
            a = pm.Dirichlet('a', a=np.ones(nGauss))
        else:
            a = np.ones(1)
        
        # Calculate distance distribution
        if nGauss==1:
            P = (1/(FWHM2sigma(w)*m.sqrt(2*m.pi)))*np.exp(-(r-r0)**2/(2*FWHM2sigma(w)**2))
        else:
            P = np.zeros(np.size(K,1))
            for i in range(nGauss):
                P += a[i]*(1/(FWHM2sigma(w[i])*m.sqrt(2*m.pi)))*np.exp(-(r-r0[i])**2/(2*FWHM2sigma(w[i])**2))
        
        # Background model
        k = pm.Gamma('k', alpha=0.5, beta=2)
        B = np.exp(-np.abs(t)*k)
        
        # DEER signal
        lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
        V0 = pm.Bound(pm.Normal,lower=0.0)('V0', mu=1, sigma=0.2)
        Vmodel = V0*(1-lamb+lamb*pm.math.dot(K,P))*B

        sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
        
        # Likelihood
        pm.Normal('V', mu=Vmodel, sigma=sigma, observed=Vdata)
        
    return model


def addData(model,V,t):

    with model:
        # Likelihood
        pm.Normal('V', mu=Vmodel, sigma=sigma, observed=Vdata)

    return model