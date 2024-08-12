import pymc as pm

import numpy as np
import deerlab as dl
import pytensor as pt
import arviz as az

from .utils import *
from .deer import *
from .samplers import *

from numpy.typing import ArrayLike

def model(
    t: ArrayLike, Vexp: ArrayLike, method: str = "regularization", 
    r: ArrayLike = None, bkgd_var: str = "Bend", n_gauss: int = 1, 
    alpha: float = None, delta_prior: tuple[float, float] = (1, 1e-6), 
    tau_prior: tuple[float, float] = (1, 1e-4)) -> dict:
    """Creates a model for sampling.
    
    The PyMC model, the DEER data, and additional parameters are all
    stored in a dictionary.

    Parameters
    ----------
    t : ArrayLike
        The time-domain axis of the DEER data.
    Vexp : ArrayLike
        The experimental signal of the DEER data.
    method : str, default="regularization"
        The method (regularization, regularization_NUTS, or gaussian) 
        to use.
    r : ArrayLike, optional
        The distance axis of the DEER data. If none given, a distance
        axis will be automatically generated.
    bkgd_var : str, default="Bend"
        The background parameterization to use. Options are "Bend" and 
        "k".
    n_gauss : int, default=1
        The number of gaussians to use. Ignored if the method is a
        regularization method.
    alpha : float, optional
        The fixed value for the regularization parameter. If not set,
        alpha will be allowed to vary.
    delta_prior : tuple of float, default=(1, 1e-6)
        The alpha and beta parameters of the gamma distribution of the
        prior for delta. Only used in the regularization models.
    tau_prior : tuple of float, default=(1, 1e-4)
        The alpha and beta parameters of the gamma distribution of the
        prior for tau. Only used in the regularization models.
    
    Returns
    -------
    model : dict
        Returns a dictionary containing the PyMC model, the DEER data,
        and additional parameters.
    """
    # Rescale data to max 1
    Vscale = np.amax(Vexp)
    Vexp_scaled = Vexp/Vscale

    # Generates distance axis if not provided
    if r is None:
        rmin = 0
        tmax = max(abs(t))
        rmax = (108*tmax)**0.333333333333333
        nr = 50
        r = np.linspace(rmin,rmax,nr)

    # Sets up model based on method
    if method == "gaussian":
        K0 = dl.dipolarkernel(t, r, integralop=True)
        model = {"t":t, "Vexp": Vexp_scaled, "r": r, "K0": K0, 
                 "n_gauss": n_gauss, "bkgd_var": bkgd_var}
        model_pymc = multigaussmodel(**model)

    elif (method == "regularization" or method == "regularization_NUTS"):
        K0 = dl.dipolarkernel(t, r, integralop=False)
        L = dl.regoperator(np.arange(len(r)), 2, includeedges=False)
        LtL = L.T@L
        K0tK0 = K0.T@K0
 
        tau_gibbs = method == "regularization"
        delta_gibbs = (method == "regularization" and alpha is None)
        all_NUTS = method == "regularization_NUTS"

        model = {"t":t, "Vexp":Vexp_scaled, "r": r, "K0": K0, "L": L, 
                 "LtL": LtL, "delta_prior": delta_prior, "tau_prior": tau_prior, 
                 "alpha": alpha, "bkgd_var": bkgd_var}
        model_pymc = regularizationmodel(tau_gibbs=tau_gibbs,
                                         delta_gibbs=delta_gibbs,
                                         all_NUTS=all_NUTS, **model)
    
    else:
        raise ValueError(f"Unknown method '{method}'.")
    
    # Update model dictionary
    model['method'] = method
    model['Vscale'] = Vscale
    model['dr'] = r[1]-r[0]
    model['model_pymc'] = model_pymc
    
    # Print information about data and model
    print(f"Time range:         {min(t):g} µs to {max(t):g} µs  ({len(t):d} points, step size {t[1]-t[0]:g} µs)")
    print(f"Distance range:     {min(r):g} nm to {max(r):g} nm  ({len(r):d} points, step size {r[1]-r[0]:g} nm)")
    print(f"Vexp max:           {Vscale:g}")
    print(f"Background:         exponential")
    if method == "gaussian":
        print(f"P model:            {n_gauss} Gaussians")
    else:
        print(f"P model:            {method}")
    
    return model

def multigaussmodel(
    t: ArrayLike, Vexp: ArrayLike, K0: np.ndarray, r: ArrayLike, 
    n_gauss: int = 1, bkgd_var: str = "Bend") -> pm.Model:
    """Generates a parametric PyMC model.
    
    The model is for a DEER signal over time vector t (in µs) given data 
    in Vexp. It uses a multi-Gaussian distribution for P with n_gauss
    gaussians and uses an exponential background.

    Model parameters:
        r0     distance(s) of mean of gaussian(s) (nm)
        w      width of gaussian(s) (nm)
        a      relative amplitudes of gaussians (when multiple)
        tau    noise precision (inverse of noise variance)
        lamb   modulation amplitude
        Bend   background decay value at end of time interval
        V0     overall amplitude

    Parameters
    ----------
    t : ArrayLike
        The time-domain axis of the DEER data.
    Vexp : ArrayLike
        The experimental signal of the DEER data.
    K0 : np.ndarray
        The dipolar kernel matrix.
    r : ArrayLike
        The distance axis of the DEER data. If none given, a distance
        axis will be automatically generated.
    n_gauss : int, default=1
        The number of gaussians to use.
    bkgd_var : str, default="Bend"
        The background parameterization to use. Options are "Bend" and
        "k".

    Returns
    -------
    model_pymc : pm.Model
        A PyMC Model object representing a parametric DEER model.
    """
    # Model definition
    with pm.Model() as model_pymc:
        # Distance distribution parameters
        r0_rel = pm.Beta('r0_rel', alpha=2, beta=2, shape=n_gauss)
        # for reporting
        r0 = pm.Deterministic('r0', r0_rel.sort()*(max(r)-min(r)) + min(r))
        w = pm.TruncatedNormal('w', pm.InverseGamma('w_mu', alpha=0.1, beta=0.2, 
                               shape=n_gauss), lower=0.05, upper=3.0, 
                               shape=n_gauss)
        if n_gauss>1:
            a = pm.Dirichlet('a', a=np.ones(n_gauss))
        # Calculate distance distribution
        if n_gauss == 1:
            P = gauss(r, r0, FWHM2sigma(w))
        else:
            P = np.zeros(np.size(K0,1))
            for i in range(n_gauss):
                P += a[i]*gauss(r, r0[i], FWHM2sigma(w[i]))
        # normalize P
        scale = P.sum() * (r[1]-r[0])
        P /= scale
        # for reporting
        pm.Deterministic('P', P)

        # Time-domain model signal
        Vmodel = pm.math.dot(K0,P)

        # Add modulation depth
        lamb = pm.Beta('lamb', alpha=1.3, beta=2.0, initval=0.2)
        Vmodel = (1-lamb) + lamb*Vmodel

        # Add background
        if bkgd_var == "k":
            k = pm.Exponential("k", scale=0.1)
            # for reporting
            Bend = pm.Deterministic("Bend", np.exp(0-k*t[-1]))
        else:
            Bend = pm.Beta("Bend", alpha=1.0, beta=1.5)
            # for reporting
            k = pm.Deterministic('k', -np.log(Bend)/t[-1])
        B = bg_exp(t,k)
        Vmodel *= B

        # Add overall amplitude
        V0 = pm.TruncatedNormal('V0', mu=1, sigma=0.2, lower=0)
        Vmodel *= V0
        
        # Noise level
        #sigma = pm.Gamma('sigma', alpha=1, beta=0.1)  # old prior
        sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
        # for reporting
        tau = pm.Deterministic("tau", 1/(sigma**2))

        # Likelihood
        pm.Normal('V', mu=Vmodel, sigma=sigma, observed=Vexp)
        
    return model_pymc

def regularizationmodel(
    t: ArrayLike, Vexp: ArrayLike, K0: np.ndarray, L: np.ndarray, 
    LtL: np.ndarray, r: ArrayLike, delta_prior: tuple[float,float] = (1, 1e-6), 
    tau_prior: tuple[float,float] = (1, 1e-4), tau_gibbs: bool = True, 
    delta_gibbs: bool = True, bkgd_var: str = "Bend", alpha: float = None, 
    all_NUTS: bool = False) -> pm.Model:
    """Generates a nonparametric PyMC model.

    The model is for a DEER signal over time vector t (in µs) given data 
    in Vexp. It uses Tikhonov regularization for P with regularizaton
    parameter alpha and uses an exponential background.

    Model parameters:
        P      distance distribution vector (nm^-1)
        tau    noise precision (inverse of noise variance)
        delta  smoothing hyperparameter (= alpha^2/sigma^2)
        lamb   modulation amplitude
        Bend   background decay value at end of time interval
        V0     overall amplitude

    Parameters
    ----------
    t : ArrayLike
        The time-domain axis of the DEER data.
    Vexp : ArrayLike
        The experimental signal of the DEER data.
    K0 : np.ndarray
        The dipolar kernel matrix.
    L : np.ndarray
        The regularization operator matrix.
    LtL : np.ndarray
        The regularization operator matrix multiplied with its 
        transpose.
    r : ArrayLike
        The distance axis of the DEER data. If none given, a distance
        axis will be automatically generated.
    delta_prior : tuple of float, default=(1, 1e-6)
        The alpha and beta parameters of the gamma distribution of the
        prior for delta. Only used in the regularization models.
    tau_prior : tuple of float, default=(1, 1e-4)
        The alpha and beta parameters of the gamma distribution of the
        prior for tau. Only used in the regularization models.
    tau_gibbs : bool, default=True
        Whether or not to sample tau with the Gibbs sampling method.
    delta_gibbs : bool, default=True
        Whether or not to sample delta with the Gibbs sampling method.
    bkgd_var : str, default="Bend"
        The background parameterization to use. Options are "Bend" and
        "k".
    alpha : float, optional
        The fixed value for the regularization parameter. If not set,
        alpha will be allowed to vary.
    all_NUTS : bool, default=False
        Whether or not to sample all parameters with the NUTS sampling
        method.

    Returns
    -------
    model_pymc : pm.Model
        A PyMC Model object representing a nonparametric DEER model.
    """
    dr = r[1]-r[0]
    
    # Model definition
    with pm.Model() as model_pymc:               
        # Noise parameter
        if tau_gibbs: # no prior (it's included in the Gibbs sampler)
            tau = pm.Flat('tau', initval=1.2)
        else:
            tau = pm.Gamma('tau', alpha=tau_prior[0], beta=tau_prior[1], 
                           initval=1.3)
        # for reporting
        sigma = pm.Deterministic('sigma', 1/np.sqrt(tau))

        # Regularization parameter
        if delta_gibbs: # no prior (it's included in the Gibbs sampler)
            delta = pm.Flat('delta', initval=1.02)
        elif alpha is not None:
            delta = pm.Deterministic('delta', alpha**2 * tau)
        else:
            # for a gamma prior on 1/delta instead of delta:
            #delta_inv = pm.Gamma('delta_inv', alpha=1, beta=1)
            #delta = pm.Deterministic('delta', 1/delta_inv)
            delta = pm.Gamma('delta', alpha=delta_prior[0], beta=delta_prior[1])
        # for reporting
        lg_alpha = pm.Deterministic('lg_alpha', np.log10(np.sqrt(delta/tau)))
        lg_delta = pm.Deterministic('lg_delta', np.log10(delta))

        # Distance distribution - no prior (it's included in the Gibbs sampler)
        if all_NUTS:
            # sums to one
            P_Dirichlet = pm.Dirichlet('P_Dirichlet', shape=len(r), 
                                       a=np.ones(len(r)))
            # integrates to one
            P = pm.Deterministic('P', P_Dirichlet/dr)
            n_p = len(np.nonzero(np.asarray(P))[0])
            # smoothness prior
            smoothness = pm.Potential("P_smoothness", 0.5*n_p*np.log(delta)
                                      -0.5*delta*np.linalg.norm(L@P)**2)
        else:
            P = pm.MvNormal('P', shape=len(r), mu=np.zeros(len(r)), 
                            cov=np.identity(len(r)))
        
        # Time-domain model signal
        Vmodel = pm.math.dot(K0*dr,P)

        # Add modulation depth
        lamb = pm.Beta('lamb', alpha=1.3, beta=2.0, initval=0.2)
        Vmodel = (1-lamb) + lamb*Vmodel

        # Add background
        if bkgd_var == "k":
            k = pm.Exponential("k", scale=0.1)
            # for reporting
            Bend = pm.Deterministic("Bend", np.exp(0-k*t[-1]))
        else:
            Bend = pm.Beta("Bend", alpha=1.0, beta=1.5)
            # for reporting
            k = pm.Deterministic('k', -np.log(Bend)/t[-1])
        B = bg_exp(t,k)
        Vmodel *= B

        # Add overall amplitude
        V0 = pm.TruncatedNormal('V0', mu=1, sigma=0.2, lower=0)
        Vmodel *= V0
        
        # Add likelihood
        pm.Normal('V', mu=Vmodel, tau=tau, observed=Vexp)
        
    return model_pymc


def sample(model: dict, **kwargs) -> az.InferenceData:
    """Samples the provided model with PyMC.

    Depending on the model method, Gibbs sampling is used for none to
    some of the parameters, and NUTS sampling is used for the rest.

    Parameters
    ----------
    model : dict
        The dictionary containing the PyMC model, model parameters, and
        DEER data from `dive.model()`.
    **kwargs : dict, optional
        Extra arguments to be passed to `pm.sample()`.
    
    Returns
    -------
    trace : az.InferenceData
        An ArviZ `InferenceData` object containing the posterior samples
        and some supplementary information.

    See Also
    --------
    pm.sample
    """
    # Important variables    
    model_pymc = model['model_pymc']
    method = model['method']
    bkgd_var = model['bkgd_var']
    alpha = model['alpha'] if 'alpha' in model else None
    
    # Set stepping methods, depending on model
    if method == "gaussian":   
        with model_pymc:
            NUTS_varlist = ['r0_rel','w','w_mu','a','sigma',bkgd_var,'V0',
                            'lamb']
            pymc_varlist = []
            for var in NUTS_varlist:
                if var in model_pymc:
                    pymc_varlist.append(model_pymc[var])
            step = [pm.NUTS(pymc_varlist, on_unused_input="ignore")]
            remove_vars  = ["r0_rel","w_mu"]
        
    elif method == "regularization":
        # Keys to pass on to sampling functions
        keys_tau = {"t","Vexp","dr","K0","tau_prior"}
        model_tau = {k: model[k] for k in keys_tau}
        keys_P = {"t","Vexp","dr","K0","LtL"}
        model_P = {k: model[k] for k in keys_P}
        model_P.update({"alpha":alpha})
        keys_delta = {"L","delta_prior"}
        model_delta = {k: model[k] for k in keys_delta}    

        with model_pymc:
            conjstep_tau = TauSampler(model_tau)
            keys_include = {}
            conjstep_P = PSampler(model_P)
            if alpha is None:
                conjstep_delta = DeltaSampler(model_delta)
            NUTS_varlist = ['V0','lamb',bkgd_var]
            pymc_varlist = []
            for var in NUTS_varlist:
                if var in model_pymc:
                    pymc_varlist.append(model_pymc[var])
            step_NUTS = pm.NUTS(pymc_varlist, on_unused_input="ignore")
        step = [conjstep_tau, conjstep_P, step_NUTS]
        if alpha is None:
            step.insert(1,conjstep_delta)
        remove_vars = ["lg_alpha"] if alpha is not None else None        
 
    elif method == "regularization_NUTS":
        step = None
        remove_vars = None
            
    else:
        raise KeyError(f"Unknown method '{method}'.",method)

    # Perform MCMC sampling
    trace = pm.sample(model=model_pymc, step=step, **kwargs)

    # Remove undesired variables
    if remove_vars is not None:
        for key in remove_vars:
            if key in trace.posterior:
                del trace.posterior[key]
                
    # Postprocessing to add in data form the model
    trace.observed_data.coords["V_dim_0"] = model["t"]
    trace.posterior.coords["P_dim_0"] = model["r"]
    trace.posterior.attrs["method"] = method
    trace.posterior.attrs["bkgd_var"] = bkgd_var
    if alpha is not None:
        trace.posterior.attrs["alpha"] = alpha
    if "delta_prior" in model:
        trace.posterior.attrs["delta_prior"] = model["delta_prior"]
    if "tau_prior" in model:
        trace.posterior.attrs["tau_prior"] = model["tau_prior"]
    if "random_seed" in kwargs.keys():
        trace.posterior.attrs["random_seed"] = kwargs["random_seed"]
    if "n_gauss" in model:
        trace.posterior.attrs["n_gauss"] = model["n_gauss"]      

    return trace
