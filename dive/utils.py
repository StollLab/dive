import numpy as np
import math as m
import sys
from scipy.special import fresnel
import pymc3 as pm
from datetime import date
import os   

from .constants import *
from .deerload import*

def addnoise(V,sig):
    """
    Add Gaussian noise with standard deviation sig to signal
    """
    noise = np.random.normal(0,sig,np.size(V))
    Vnoisy = V + noise
    return Vnoisy

def dipolarkernel(t,r):
    """
    K = dipolarkernel(t,r)
    Calculate dipolar kernel matrix.
    Assumes t in microseconds and r in nanometers
    """
    wr = w0/(r*1e-9)**3  # rad s^-1
        
    # Calculation using Fresnel integrals
    nr = np.size(r)
    nt = np.size(t)
    K = np.zeros((nt,nr))
    for ir in range(nr):
        ph = wr[ir]*(np.abs(t)*1e-6)
        kappa = np.sqrt(6*ph/m.pi)
        S, C = fresnel(kappa)
        K[:,ir] = (np.cos(ph)*C+np.sin(ph)*S)/kappa
    
    K[t==0,:] = 1   # fix div by zero
    
    # Include delta-r factor for integration
    if len(r)>1:
        dr = np.mean(np.diff(r))
        K = K*dr
    
    return K

def loadTrace(FileName):
    """
    Load a DEER trace, can be a bruker or comma separated file.
    """
    if FileName.endswith('.dat') or FileName.endswith('.txt') or FileName.endswith('.csv'):
        data = np.genfromtxt(FileName,delimiter=',',skip_header=1)
        t = data[:,0]
        Vdata = data[:,1]
    elif FileName.endswith('.dat'):
        t, Vdata, Parameters = deerload(FileName)
    else:
        sys.exit('The file format is not recognized.')

    return t, Vdata

def sample(Model,MCMCparameters):
    """ 
    Use pymc3 to draw samples from the posterior for the model, according to the parameters provided with MCMCparameters.
    """
    RequiredKeys = ["draws","tune","chains"]

    for Key in RequiredKeys:
        if Key not in MCMCparameters:
            sys.exit("{} is a required MCMC parameter".format(Key))            

    if "cores" not in MCMCparameters:
        MCMCparameters["cores"] = 2
    
    # Sampling
    trace = pm.sample(model=Model, draws=MCMCparameters["draws"], tune=MCMCparameters["tune"], chains=MCMCparameters["chains"],cores=MCMCparameters["cores"])

    # # Result processing 
    df = pm.trace_to_dataframe(trace)

    return df

def saveTrace(df,Parameters,SaveName='empty'):
    
    if SaveName == 'empty':
        today = date.today()
        datestring = today.strftime("%Y%m%d")
        SaveName = "./traces/{}_traces.dat".format(datestring)
    
    if not SaveName.endswith('.dat'):
        SaveName = SaveName+'.dat'

    shape = df.shape 
    cols = df.columns.tolist()

    os.makedirs(os.path.dirname(SaveName), exist_ok=True)

    f = open(SaveName, 'a+')
    f.write("# Traces from the MCMC simulations with pymc3\n")
    f.write("# The following {} parameters were investigated:\n".format(shape[1]))
    f.write("# {}\n".format(cols))
    f.write("# nParameters nChains nIterations\n")
    if Parameters['nGauss'] == 1:
        f.write("{},{},{},0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 2:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 3:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 4:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))

    df.to_csv (f, index=False, header=False)

    f.close()
