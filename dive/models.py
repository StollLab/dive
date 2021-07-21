import pymc3 as pm
import numpy as np
import math as m
import deerlab as dl

from .utils import *
from .deer import *

from theano import tensor as T
from theano.tensor import nlinalg as tnp
from theano.tensor import slinalg as snp

def model(t, Vdata, pars):

    allowed_methods = ['regularization', 'gaussian', 'regularization_taubased']

    # Rescale data to max 1
    Vscale = np.amax(Vdata)
    Vdata /= Vscale

    if 'method' not in pars:
        sys.exit("'method' is a required field")
    elif pars['method'] not in allowed_methods:
        sys.exit("keyword 'method' has to be one of the following: " + ', '.join(allowed_methods))

    if pars['method'] == 'gaussian':
        if 'nGauss' not in pars:
           sys.exit("nGauss is a required key for 'method' " + pars['method']) 

        # Calculate dipolar kernel for integration
        r = np.linspace(1,10,451)
        K0 = dl.dipolarkernel(t,r)
        model_graph = multigaussmodel(pars['nGauss'], t, Vdata, K0, r)
        model_pars = {"r": r, "ngaussians" : pars['nGauss']}

    # elif pars['method'] == 'regularization':
    #     if 'r' not in pars:
    #        sys.exit("r is a required key for 'method' = " + pars['method']) 

    #     a_delta = 1
    #     b_delta = 1e-6   

    #     K0 = dl.dipolarkernel(t,pars['r'],integralop=False)
    #     model_graph = regularizationmodel(t, Vdata, K0, pars['r'], a_delta, b_delta)

    #     K0 = dl.dipolarkernel(t,pars['r'],integralop=False)   # kernel matrix
    #     L = dl.regoperator(np.linspace(1,len(pars['r']),len(pars['r'])), 2)
    #     LtL = np.matmul(np.transpose(L),L)
    #     K0tK0 = np.matmul(np.transpose(K0),K0)

    #     model_pars = {'K0': K0, 'L': L, 'LtL': LtL, 'K0tK0': K0tK0, "r": pars['r'], 'a_delta': a_delta, 'b_delta': b_delta}

    elif pars['method'] == 'regularization':
        if 'r' not in pars:
           sys.exit("r is a required key for 'method' = " + pars['method']) 

        a_delta = 1
        b_delta = 1e-6

        a_tau = 1
        b_tau = 1e-4  

        K0 = dl.dipolarkernel(t,pars['r'],integralop=False)
        model_graph = regularizationmodel(t, Vdata, K0, pars['r'], a_delta, b_delta, a_tau, b_tau)

        K0 = dl.dipolarkernel(t,pars['r'],integralop=False)   # kernel matrix
        L = dl.regoperator(np.linspace(1,len(pars['r']),len(pars['r'])), 2)
        LtL = np.matmul(np.transpose(L),L)
        K0tK0 = np.matmul(np.transpose(K0),K0)

        model_pars = {'K0': K0, 'L': L, 'LtL': LtL, 'K0tK0': K0tK0, "r": pars['r'], 'a_delta': a_delta, 'b_delta': b_delta, 'a_tau': a_tau, 'b_tau': b_tau}
    
    model_pars['method'] = pars['method']
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
        V0 = pm.Bound(pm.Normal,lower=0.0)('V0', mu=1, sigma=0.2)
        
        Vmodel = deerTrace(pm.math.dot(K0,P),B,V0,lamb)

        sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
        
        # Likelihood
        pm.Normal('V', mu = Vmodel, sigma = sigma, observed = Vdata)
        
    return model

# def regularizationmodel(t, Vdata, K0, r, a_delta, b_delta):
#     """
#     Generates a PyMC3 model for a DEER signal over time vector t
#     (in microseconds) given data in Vdata.
#     """ 

#     dr = r[1] - r[0]
    
#     # Model definition
#     model = pm.Model()
#     with model:
#         # Noise --------------------------------------------------------------
#         sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
#         tau = pm.Deterministic('tau',1/(sigma**2))

#         # Regularization parameter -------------------------------------------
#         delta = pm.Uniform('delta', lower= 0, upper = 1e20, transform = None)
#         lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)) )
        
#         # Time Domain --------------------------------------------------------
#         lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
#         V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1, sigma=0.2)

#         # Background ---------------------------------------------------------
#         k = pm.Gamma('k', alpha=0.5, beta=2)
#         B = dl.bg_exp(t, k)

#         # Distance distribution ----------------------------------------------
#         P = pm.Uniform("P", lower= 0, upper = 1000, shape = len(r), transform = None)      

#         # Calculate matrices and operators -----------------------------------
#         Kintra = (1-lamb) + lamb*K0
#         B_ = T.transpose( T.tile(B,(len(r),1)) )
#         K = V0*Kintra*B_*dr

#         # Time domain model ---------------------------------------------------
#         Vmodel = pm.math.dot(K,P)

#         # Likelihood ----------------------------------------------------------
#         pm.Normal('V',mu = Vmodel, sigma = sigma, observed = Vdata)
        
#     return model

def regularizationmodel(t, Vdata, K0, r, a_delta, b_delta, a_tau, b_tau):
    """
    Generates a PyMC3 model for a DEER signal over time vector t
    (in microseconds) given data in Vdata.
    """ 

    dr = r[1] - r[0]
    
    # Model definition
    model = pm.Model()
    with model:
        # Noise --------------------------------------------------------------
        tau = pm.Uniform('tau', lower= 0, upper = 1e20, transform = None)
        sigma = pm.Deterministic('sigma',1/np.sqrt(tau))

        # Regularization parameter -------------------------------------------
        delta = pm.Uniform('delta', lower= 0, upper = 1e20, transform = None)
        lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)) )
        
        # Time Domain --------------------------------------------------------
        lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
        V0 = pm.Bound(pm.Normal, lower=0.0)('V0', mu=1, sigma=0.2)

        # Background ---------------------------------------------------------
        k = pm.Gamma('k', alpha=0.5, beta=2)
        B = dl.bg_exp(t, k)

        # Distance distribution ----------------------------------------------
        P = pm.Uniform("P", lower= 0, upper = 1000, shape = len(r), transform = None)     

        # Calculate matrices and operators -----------------------------------
        Kintra = (1-lamb) + lamb*K0
        B_ = T.transpose( T.tile(B,(len(r),1)) )
        K = V0*Kintra*B_*dr

        # Time domain model ---------------------------------------------------
        Vmodel = pm.math.dot(K,P)

        # Likelihood ----------------------------------------------------------
        pm.Normal('V',mu = Vmodel, sigma = sigma, observed = Vdata)
        
    return model