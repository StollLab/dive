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
    trace, model_return : az.InferenceData, dict
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
    attributes = ["method","bkgd_var","alpha","include_background",
                  "include_mod_depth","include_amplitude","delta_prior"
                  "tau_prior","n_gauss"]
    attr_dict = {}
    for attr in attributes:
        if attr in trace.posterior.attrs:
            attr_dict.update({attr:trace.posterior.attrs[attr]})
    # background was renamed to bkgd_var
    if "background" in trace.posterior.attrs:
        attr_dict.update({"bkgd_var":trace.posterior.attrs["background"]})
    # convert n_gauss back to integer
    if "n_gauss" in attr_dict:
        attr_dict.update({"n_gauss":int(attr_dict["n_gauss"])})

    model_return = model(t, Vexp, r=r, **attr_dict)
    return trace, model_return