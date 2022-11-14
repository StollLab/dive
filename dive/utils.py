from dataclasses import replace
from random import seed
import numpy as np
import math as m
import sys
from scipy.special import fresnel
import pymc3 as pm
from datetime import date
import os   
import copy
import random 
from .constants import *
from .deerload import *
from .samplers import *
import arviz as az
from .plotting import *

def addnoise(V,sig):

    """
    Add Gaussian noise with standard deviation sig to signal
    """
    noise = np.random.normal(0, sig, np.size(V))
    Vnoisy = V + noise
    return Vnoisy


def FWHM2sigma(FWHM):
    """
    Convert the full width at half maximum, FWHM, of a Gaussian to the standard deviation, sigma.
    """

    sigma = FWHM/(2*m.sqrt(2*m.log(2)))

    return sigma


def sigma2FWHM(sigma):
    """
    Convert the standard deviation, sigma, of a Gaussian to the full width at half maximum, FWHM.
    """
    FWHM = sigma/(2*m.sqrt(2*m.log(2)))

    return FWHM


def dipolarkernel(t,r):
    """
    K = dipolarkernel(t,r)
    Calculate dipolar kernel matrix.
    Assumes t in microseconds and r in nanometers
    """
    omega = 1e-6 * D/(r*1e-9)**3  # rad µs^-1
    
    # Calculation using Fresnel integrals
    nr = np.size(r)
    nt = np.size(t)
    K = np.zeros((nt, nr))
    for ir in range(nr):
        ph = omega[ir]*np.abs(t)
        z = np.sqrt(6*ph/m.pi)
        S, C = fresnel(z)
        K[:,ir] = (C*np.cos(ph)+S*np.sin(ph))/z
    
    K[t==0,:] = 1  # fix div by zero
    
    # Include delta-r factor for integration
    if len(r)>1:
        dr = np.mean(np.diff(r))
        K *= dr
    
    return K


def loadTrace(FileName):
    """
    Load a DEER trace, can be a Bruker or comma separated file.
    """
    if FileName.endswith('.dat') or FileName.endswith('.txt') or FileName.endswith('.csv'):
        data = np.genfromtxt(FileName, delimiter=',', skip_header=1)
        t = data[:,0]
        Vdata = data[:,1]
    elif FileName.endswith('.DTA') or FileName.endswith('.DSC'):
        t, Vdata, Parameters = deerload(FileName)
    else:
        raise ValueError('The file format is not recognized.')

    return t, Vdata


def sample(model_dic, MCMCparameters, steporder=None, NUTSorder=None, NUTSpars=None):
    """
    Use PyMC3 to draw samples from the posterior for the model, according to the parameters provided with MCMCparameters.
    """
    
    # Complain about missing required keywords
    requiredKeys = ["draws", "tune", "chains"]
    for key in requiredKeys:
        if key not in MCMCparameters:
            raise KeyError(f"The required MCMC parameter '{key}' is missing.")
    
    # Supplement defaults for optional keywords
    defaults = {"cores": 2, "progressbar": True, "return_inferencedata": False}
    MCMCparameters = {**defaults, **MCMCparameters}
    
    model = model_dic['model']
    model_pars = model_dic['pars']
    method = model_pars['method']
    bkgd_var = model_pars['bkgd_var']
    
    # Set stepping methods, depending on model
    if method == "gaussian":
        
        removeVars  = ["r0_rel"]
        
        with model:
            if model_pars['ngaussians']==1:
                if bkgd_var=="k":
                    NUTS_varlist = [model['r0_rel'], model['w'], model['sigma'], model['k'], model['V0'], model['lamb']]
                elif bkgd_var=="tauB":
                    NUTS_varlist = [model['r0_rel'], model['w'], model['sigma'], model['tauB'], model['V0'], model['lamb']]
                elif bkgd_var=="Bend":
                    NUTS_varlist = [model['r0_rel'], model['w'], model['sigma'], model['Bend'], model['V0'], model['lamb']]
            else:
                if bkgd_var=="k":
                    NUTS_varlist = [model['r0_rel'], model['w'], model['a'], model['sigma'], model['k'], model['V0'], model['lamb']]
                elif bkgd_var=="tauB":
                    NUTS_varlist = [model['r0_rel'], model['w'], model['a'], model['sigma'], model['tauB'], model['V0'], model['lamb']]
                elif bkgd_var=="Bend":
                    NUTS_varlist = [model['r0_rel'], model['w'], model['a'], model['sigma'], model['Bend'], model['V0'], model['lamb']]
            if NUTSorder is not None:
                NUTS_varlist = [NUTS_varlist[i] for i in NUTSorder] 
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
    
        step = [step_NUTS]
        
    elif method == "regularization":
        
        removeVars = None
        
        with model:
            if bkgd_var=="k":
                NUTS_varlist = [model['k'], model['V0'], model['lamb']]
                step_tau = randTau_k_posterior(model['tau'], model_pars['tau_prior'], model_pars['K0'], model['P'], model_dic['Vexp'], model_pars['r'], model_dic['t'], model['k'], model['lamb'], model['V0'])
                step_P = randPnorm_k_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['k'], model['lamb'], model['V0'])
            elif bkgd_var=="tauB":
                NUTS_varlist = [model['tauB'], model['V0'], model['lamb']]
                step_tau = randTau_tauB_posterior(model['tau'], model_pars['tau_prior'], model_pars['K0'], model['P'], model_dic['Vexp'], model_pars['r'], model_dic['t'], model['tauB'], model['lamb'], model['V0'])
                step_P = randPnorm_tauB_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['tauB'], model['lamb'], model['V0'])
            else:
                NUTS_varlist = [model['Bend'], model['V0'], model['lamb']]
                step_tau = randTau_Bend_posterior(model['tau'], model_pars['tau_prior'], model_pars['K0'], model['P'], model_dic['Vexp'], model_pars['r'], model_dic['t'], model['Bend'], model['lamb'], model['V0'])
                step_P = randPnorm_Bend_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['Bend'], model['lamb'], model['V0'])
            
            if NUTSorder is not None:
                NUTS_varlist = [NUTS_varlist[i] for i in NUTSorder] 
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
            
            step_delta = randDelta_posterior(model['delta'], model_pars['delta_prior'], model_pars['L'], model['P'])

        # Original order
        #step = [step_P, step_tau, step_delta, step_NUTS]

        # As written in manuscript 
        step = [step_tau, step_delta, step_P, step_NUTS]
        if steporder is not None:
            step = [step[i] for i in steporder]
                
    elif method == "regularization2":
        
        removeVars = None
        
        with model:
            if bkgd_var=="k":
                NUTS_varlist = [model['tau'], model['delta'], model['k'], model['V0'], model['lamb']]
                step_P = randPnorm_k_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['k'], model['lamb'], model['V0'])
            elif bkgd_var=="tauB":
                NUTS_varlist = [model['tau'], model['delta'], model['tauB'], model['V0'], model['lamb']]
                step_P = randPnorm_tauB_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['tauB'], model['lamb'], model['V0'])
            else:
                NUTS_varlist = [model['tau'], model['delta'], model['Bend'], model['V0'], model['lamb']]
                step_P = randPnorm_Bend_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['Bend'], model['lamb'], model['V0'])
                
            if NUTSorder is not None:
                NUTS_varlist = [NUTS_varlist[i] for i in NUTSorder] 
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
        
        step = [step_P, step_NUTS]
        if steporder is not None:
            step = [step[i] for i in steporder]
                
    else:
        
        raise KeyError(f"Unknown method '{method}'.",method)

    # Perform MCMC sampling

    #if seed is not None:
    #    trace = pm.sample(model=model, step=step, random_seed=seed,  **MCMCparameters)
    #else: 
    trace = pm.sample(model=model, step=step,  **MCMCparameters)




    # Remove undesired variables
    if removeVars is not None:
        [trace.remove_values(key) for key in removeVars if key in trace.varnames]

    return trace




        
def interpret(trace,model_dic):
    
    
    class FitResult:
        def __init__(self,trace, model):
            d = {key: trace[key] for key in trace.varnames}
            self.__dict__.update(d)

            self.r = model['pars']['r']
            self.t = model['t']
            self.Vexp = model['Vexp']
            self.varnames = trace.varnames
            self.trace = trace
            self.K = dl.dipolarkernel(self.t, self.r)
            self.dr = self.r[1] - self.r[0]

            # self.plots = Plots(trace,model)

        def subsample_fits(self, n=100, seed=1):
            np.random.seed(seed)
            idxs = np.random.choice(len(self.trace), n, replace=False)
            Ps = [self.P[idx].copy() for idx in idxs]
            Bs, Vs = [], []


            for idx in idxs:
                V_ = self.K@self.P[idx]

                if 'lamb' in self.varnames:
                    V_ = (1-self.lamb[idx]) + self.lamb[idx]*V_

                if 'k' in self.varnames:
                    B = dl.bg_exp(self.t, self.k[idx])
                    V_ *= B
                    
                    Blamb = (1-self.lamb[idx])*B
                
                if 'V0' in self.varnames:
                    Blamb *= self.V0[idx]
                    Bs.append(Blamb)
                    V_ *= self.V0[idx]
                Vs.append(V_)

            return Vs, Bs, Ps

        def plot(self,style = 'noodle',j =0.95):
            plt.style.use('seaborn-darkgrid')
            
            plt.rcParams.update({'font.family':'serif'})
            if style == 'noodle':
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))
                Vs, Bs, Ps = self.subsample_fits()
                
                ax1.plot(self.t, self.Vexp,'g.',linewidth=0.5,alpha = 0.3)

                for V, B, P in zip(Vs, Bs, Ps):
                    ax2.plot(self.r, P, 'cornflowerblue', linewidth=0.3)
                    ax1.plot(self.t, V,'#0000EE',linewidth=0.3)
                    ax1.plot(self.t, B,'#FAD02C',linewidth=0.3)
                    ax1.plot(self.t, V-self.Vexp,'#FF0080',linewidth=0.3)

                leg1= ax1.legend(['Data','Vexp','Background','Residuals'])
                leg2 = ax2.legend(['Distance Distribution'])

                for lh1,lh2 in zip(leg1.legendHandles,leg2.legendHandles): 
                    lh1.set_alpha(1)
                    lh2.set_alpha(1)
                

                #ax2.set_xlabel(r'time ($\rm\mus$)')

            if style == 'mean-ci':
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))
                Vs, Bs, Ps = self.subsample_fits()
                
                l0, = ax1.plot(self.t, self.Vexp,'#808080',marker='.',linewidth=0.5,alpha = 0.3,label = 'Data',linestyle='None')

                Pmean = np.mean(Ps,0)
                Phd = az.hdi(np.array(Ps),j) 

                Vmean = np.mean(Vs,0)
                Vhd = az.hdi(np.array(Vs),j) 
                

                Bmean = np.mean(Bs,0)
                Bhd = az.hdi(np.array(Bs),j) 

            
                ax2.plot(self.r, Pmean, '#8E05D4', linewidth=1)
                ax2.fill_between(self.r,Phd[:,0],Phd[:,1],alpha = 0.7)


                l1,=ax1.plot(self.t, Vmean, '#964B00', linewidth=0.5,label='Vexp mean')
                l2,=ax1.plot(self.t, Bmean, 'b', linewidth=0.5 ,label='Background mean')
                l3,=ax1.plot(self.t, Vmean-self.Vexp,'#FF0080',linewidth=0.8,label = "Residuals")


                ax1.fill_between(self.t,Vhd[:,0],Vhd[:,1],color = 'C0',alpha =0.5)
                ax1.fill_between(self.t,Bhd[:,0],Bhd[:,1],color = '#FFF68F',alpha =0.1)
                ax1.legend(handles=[l0,l1,l2,l3])
                ax2.legend()
                
                
                #ax2.set_xlabel(r'time ($\rm\mus$)')
            
            

            
            ax2.set_xlabel('Distance(nm)')
            ax2.set_ylabel("Probability($1/nm$)")
            ax2.xaxis.set_major_locator(plt.MaxNLocator(16))


            ax1.set_ylabel('Signal (a.u.)')
            ax1.set_xlabel("Time(µs)")
            ax1.xaxis.set_major_locator(plt.MaxNLocator(15))
           


            fig.tight_layout()
            plt.style.use('seaborn-darkgrid')
            return fig

        def summary(self):
            printsummary(self.trace,self.model)
            
            
            
    fit = FitResult(trace,model_dic)

    return fit

def saveTrace(df, Parameters, SaveName='empty'):
        """
        Save a trace to a CSV file.
        """
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

def test():
    pass