import numpy as np
import arviz as az
from datetime import date
from .models import *

def saveTrace(trace, model_dic, SaveName=None):
    """
    Saves a trace to a netCDF file.
    """
    # creates the proper name for the file
    if not SaveName:
        today = date.today()
        SaveName = 'data/' + today.strftime("%Y%m%d")

    if not SaveName.endswith('.nc'):
        SaveName = SaveName + '.nc'

    # saves the trace as a netCDF file
    trace.to_netcdf(SaveName)
    
    return

def loadTrace(path):
    """
    Returns the trace and the model dictionary from a netCDF file.
    """
    # reads netCDF file (as an InferenceData object)
    trace = az.from_netcdf(path)

    # recreates model_dic object
    t = trace.observed_data.coords["V_dim_0"].values
    Vexp = trace.observed_data["V"].values
    pars = {"method": trace.posterior.attrs["method"], "r": trace.posterior.coords["P_dim_0"].values, "background": trace.posterior.attrs["background"]}
    if "nGauss" in trace.posterior.attrs:
        pars.update({"nGauss": int(trace.posterior.attrs["nGauss"])})
    if "alpha" in trace.posterior.attrs:
        pars.update({"alpha": trace.posterior.attrs["alpha"]})

    model_dic = model(t, Vexp, pars)

    return trace, model_dic