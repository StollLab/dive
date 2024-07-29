import numpy as np
import arviz as az
from datetime import date
from .models import *

def save_trace(trace : az.InferenceData, filename: str = None):
    """Saves a trace to a netCDF file.

    If no name is provided, the date and time will be used as the 
    filename.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be saved.
    filename : str, optional
        The name of the generated file.

    See Also
    --------
    load_trace
    """
    # creates the proper name for the file
    if not filename:
        today = date.today()
        filename = 'data/' + today.strftime("%Y%m%d")
    if not filename.endswith('.nc'):
        filename = filename + '.nc'

    # saves the trace as a netCDF file
    trace.to_netcdf(filename)
    return

def load_trace(filename : str) -> tuple[az.InferenceData,dict]:
    """Returns the trace and the model dictionary from a netCDF file.

    Parameters
    ----------
    filename : str
        The filepath of the file to be read.

    Returns
    -------
    trace, model : az.InferenceData, dict
        A tuple containing the loaded trace and the recreated model
        dictionary.
    
    See Also
    --------
    save_trace
    """
    # reads netCDF file (as an InferenceData object)
    trace = az.from_netcdf(filename)

    # recreates model object
    t = trace.observed_data.coords["V_dim_0"].values
    Vexp = trace.observed_data["V"].values
    r = trace.posterior.coords["P_dim_0"].values
    method = trace.posterior.attrs["method"]
    bkgd_var = trace.posterior.attrs["bkgd_var"]
    alpha = trace.posterior.attrs["alpha"]
    include_background = trace.posterior.attrs["include_background"]
    include_mod_depth = trace.posterior.attrs["include_mod_depth"]
    include_amplitude = trace.posterior.attrs["include_amplitude"]
    delta_prior = trace.posterior.attrs["delta_prior"]
    tau_prior = trace.posterior.attrs["tau_prior"]
    n_gauss = 1
    if "n_gauss" in trace.posterior.attrs:
        n_gauss = trace.posterior.attrs["n_gauss"]

    model = model(t, Vexp, method, r, n_gauss, alpha, delta_prior, tau_prior,
                  include_background, include_mod_depth, include_amplitude)

    return trace, model