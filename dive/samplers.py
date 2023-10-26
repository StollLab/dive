import numpy as np
from scipy.linalg import sqrtm
from pymc.step_methods.arraystep import BlockedStep
import pymc as pm

from .deer import *
from .utils import fnnls

class randPnorm_posterior(BlockedStep):
    """
    Draws independent samples of P (with non-negative elements)
    from the full conditional (posterior) distribution of P.
    The returned P is normalized.
    
    Initalization needs the following
      K0    kernel matrix
      LtL   L'*L, where L is the regularization matrix
      t     time vector
      V     data vector
      r     distance vector
      
    The following model parameter are needed as well and are 
    pulled from the PyMC model context:
      tau     precision parameter
      delta   regularization parameter
      Bend    end value of background decay function (at t[-1])
      lamb    modulation depth
      V0      overall amplitude
    """
    
    def __init__(self, pars):
        # Set self.vars with the list of variables covered by this sampler
        pymcmodel = pm.modelcontext(None)
        P_value_var = pymcmodel.rvs_to_values[pymcmodel['P']]
        self.vars = [P_value_var]

        # Store data and constants
        self.V = pars["Vexp"]
        self.t = pars["t"]
        self.K0 = pars["K0"]
        self.LtL = pars["LtL"]
        self.dr = pars["dr"]

    def step(self, point: dict):
        
        # Get current parameter values and backtransform if necessary
        tau = point['tau'] if 'tau' in point else np.exp(point['tau_log__'])
        delta = point['delta'] if 'delta' in point else np.exp(point['delta_log__'])
        Bend = 1/(1+np.exp(-point['Bend_logodds__']))
        lamb = 1/(1+np.exp(-point['lamb_logodds__']))
        V0 = np.exp(point['V0_interval__'])

        # Calculate full kernel matrix
        K = (1-lamb) + lamb*self.K0
        k = -1/self.t[-1]*np.log(Bend)
        B = bg_exp(self.t,k) 
        K *= B[:, np.newaxis]
        K *= V0*self.dr

        # Calculate posterior distribution parameters
        KtK = np.matmul(np.transpose(K), K)
        KtV = np.matmul(np.transpose(K), self.V) 
        tauKtV = tau*KtV
        invSigma = tau*KtK + delta*self.LtL

        # Draw new sample of P
        Pdraw = _randP(tauKtV, invSigma)
        
        # Normalize P
        Pdraw =  Pdraw / np.sum(Pdraw) / self.dr

        # Store new sample
        newpoint = point.copy()
        newpoint['P'] = Pdraw
        
        stats = []
        return newpoint, stats

class randDelta_posterior(BlockedStep):
    
    def __init__(self, pars):
        # Set self.vars with the list of variables covered by this sampler
        model = pm.modelcontext(None)
        delta_value_var = model.rvs_to_values[model['delta']]
        self.vars = [delta_value_var]
        
        # Store constants
        delta_prior = pars["delta_prior"]
        self.a_delta = delta_prior[0]
        self.b_delta = delta_prior[1]
        self.L = pars["L"]

    def step(self, point: dict):
        
        # Get current P
        P = point['P']

        # Calculate posterior distribution parameters
        n_p = sum(np.asarray(P)>0)
        a_delta = self.a_delta + 0.5*n_p
        b_delta = self.b_delta + 0.5*np.linalg.norm(self.L@P)**2

        # Draw new sample of delta
        delta_draw = np.random.gamma(a_delta, 1/b_delta)
        
        # Save sample
        newpoint = point.copy()
        newpoint['delta'] = delta_draw
        
        stats = []
        return newpoint, stats

class randTau_posterior(BlockedStep):
    """
    based on:
    J.M. Bardsley, P.C. Hansen, MCMC Algorithms for Computational UQ of 
    Nonnegativity Constrained Linear Inverse Problems, 
    SIAM Journal on Scientific Computing 42 (2020) A1269-A1288 
    from "Hierarchical Gibbs Sampler" block after Eqn. (2.8)
    """
    def __init__(self, pars):
        # Set self.vars with the list of variables covered by this sampler
        model = pm.modelcontext(None)
        tau_value_var = model.rvs_to_values[model['tau']]
        self.vars = [tau_value_var]
        
        # Store data and constants
        self.V = pars["Vexp"]
        self.t = pars["t"]
        tau_prior = pars["tau_prior"]
        dr = pars["dr"]
        self.a_tau = tau_prior[0]
        self.b_tau = tau_prior[1]
        self.K0dr = pars["K0"]*dr

    def step(self, point: dict):

        # Get current parameter values and backtransform if necessary
        P = point['P']
        Bend = 1/(1+np.exp(-point['Bend_logodds__']))
        lamb = 1/(1+np.exp(-point['lamb_logodds__']))
        V0 = np.exp(point['V0_interval__'])

        # Calculate V model signal (without noise)
        Vmodel = (1-lamb) + lamb*(self.K0dr@P)
        k = -1/self.t[-1]*np.log(Bend)
        B = bg_exp(self.t, k) 
        Vmodel *= B
        Vmodel *= V0

        # Calculate distribution parameters
        M = len(self.V)
        a_tau = self.a_tau + 0.5*M
        b_tau = self.b_tau + 0.5*np.linalg.norm((Vmodel-self.V))**2

        # Draw new sample of tau
        tau_draw = np.random.gamma(a_tau, 1/b_tau)

        # Save new sample
        newpoint = point.copy()
        newpoint['tau'] = tau_draw
        
        stats = []
        return newpoint, stats

def _randP(tauKtX, invSigma):
    r"""
    Draws a random P with non-negative elements
    
    based on:
    J.M. Bardsley, C. Fox, An MCMC method for uncertainty quantification in
    nonnegativity constrained inverse problems, Inverse Probl. Sci. Eng. 20 (2012)
    https://doi.org/10.1080/17415977.2011.637208
    """
    Sigma = np.linalg.inv(invSigma)

    try:
        C_L = np.linalg.cholesky(Sigma)
    except:
        C_L = sqrtm(Sigma)
    
    v = np.random.standard_normal(size=(len(tauKtX),))
    w = np.linalg.solve(np.matrix.transpose(C_L), v)
    
    P = fnnls(invSigma, tauKtX+w)
    
    return P`
`