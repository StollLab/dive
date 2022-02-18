## Plotting 

# Import modules
import matplotlib.pyplot as plt
import numpy as np
import random
import arviz as az
from IPython.display import display
import deerlab as dl
import copy
from scipy.io import loadmat

from .utils import *
from .deer import *


def _relevantVariables(trace):
    desiredVars = ["r0", "w", "a", "k", "lamb", "V0", "sigma", "lg_alpha"]
    Vars = [Var for Var in desiredVars if Var in trace.varnames]
    return Vars


def printsummary(trace, model_dic):
    """
    Print table of all parameters, including their means, standard deviations,
    effective sample sizes, Monte Carlo standard errors, and R-hat diagnostics.
    """
    Vars = _relevantVariables(trace)
    with model_dic['model']:
        summary = az.summary(trace, var_names=Vars)
    # replace the labels with their unicode characters before displaying
    summary.index = _betterLabels(summary.index.values)
    display(summary)


def plotmarginals(trace, GroundTruth=None):
    """
    Plot marginalized posteriors
    """
    Vars = _relevantVariables(trace)
    nVars = len(Vars)

    # figure out layout of plots and create figure
    nCols = min(nVars,6)
    nRows = int(np.ceil(nVars/nCols))
    fig, axs = plt.subplots(nRows, nCols)
    axs = axs.flatten()
    width = min(3*nVars,12)
    height = nRows*3.5

    # set figure size
    fig.set_figheight(height)
    fig.set_figwidth(width)
    
    # KDE of chain samples and plot them
    for i in range(nVars):
        az.plot_kde(trace[Vars[i]], ax=axs[i])
        axs[i].set_xlabel(_betterLabels(Vars[i]), fontsize='large')
        axs[i].yaxis.set_ticks([])
        axs[i].grid(axis='x')

        if GroundTruth:
            if Vars[i] in GroundTruth.keys():
                bottom, top = axs[i].get_ylim()
                axs[i].vlines(GroundTruth[Vars[i]], bottom, top, color='black')

    for i in range(nVars, len(axs)):
        axs[i].axis('off')

    # Clean up figure
    fig.tight_layout()
    return fig


def plotcorrelations(trace, model_dic, figsize=None, marginals=True):
    """
    Matrix of pairwise correlation plots between model parameters.
    """
    # determine variables to include
    Vars = _relevantVariables(trace)
    nVars = len(Vars)
    
    # Set default figure size
    if figsize is None:
        if nVars < 3:
            figsize = (7, 7)
        else:
            figsize = (10, 10)

    # use arviz library to plot correlations
    with model_dic["model"]:
        axs = az.plot_pair(trace, var_names=Vars, kind='kde', figsize=figsize, marginals=marginals)

    # replace labels with the nicer unicode character versions
    if len(Vars) > 2:
        # reshape axes so that we can loop through them
        axs = np.reshape(axs,np.shape(axs)[0]*np.shape(axs)[1])

        for ax in axs:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel:
                ax.set_xlabel(_betterLabels(xlabel))
            if ylabel:
                ax.set_ylabel(_betterLabels(ylabel))
    else:
        xlabel = axs.get_xlabel()
        ylabel = axs.get_ylabel()
        axs.set_xlabel(_betterLabels(xlabel))
        axs.set_ylabel(_betterLabels(ylabel))

    fig = plt.gcf()
    return fig


def summary(trace, model_dic):
    
    printsummary(trace, model_dic)
    plotmarginals(trace)
    plotcorrelations(trace, model_dic)
    plotresult(trace, model_dic)


def plotresult(trace, model_dic, nDraws=100, Pid=None, Pref=None, rref=None, show_ave=None):
    """
    Plot the MCMC results in the time domain and in the distance domain, using an
    ensemble of P draws from the posterior, and the associated time-domain signals.
    Also shown in the time domain: the ensemble of residual vectors, and the ensemble
    of backgrounds.
    """

    if show_ave is not None:
        print('Showing posterior average')
        
         
    
    if show_ave is None:
        print('Posterior average hidden')
    
        

    
    # Get reference distribution if specified ------------------------------------
    if Pid is not None:
        P0s = loadmat('..\..\data\edwards_testset\distributions_2LZM.mat')['P0']
        rref = np.squeeze(loadmat('..\..\data\edwards_testset\distributions_2LZM.mat')['r0'])
        Pref = P0s[Pid-1,:]
        
    elif Pref is not None:
        if rref is None:
            raise KeyError("If 'Pref' is provided, 'rref' must be provided as well.")


    Vexp = model_dic['Vexp']
    t = model_dic['t']
    r = model_dic['pars']['r']
    

   
    Ps, Vs, Bs, _, _ = drawPosteriorSamples(trace, nDraws, r, t)
    fig = plotMCMC(Ps, Vs, Bs, Vexp, t, r, Pref, rref,show_ave)
    

    return fig

# Look-up table that maps variable strings to better symbols for printing
_table = {
    "k": "$k$",
    "lamb": "$λ$",
    "lamba": "$λ$",
    "sigma": "$σ$",
    "delta": "$δ$",
    "tau": "$τ$",
    "V0": "$V_0$",
    "r0": "$r_0$",
    "alpha": "$α$",
    "lg_alpha": "$\mathrm{lg}(α)$",
    "w[0]": "$w_0$",
    "w[1]": "$w_1$",
    "w[2]": "$w_2$",
    "w[3]": "$w_3$",
    "a[0]": "$a_0$",
    "a[1]": "$a_1$",
    "a[2]": "$a_2$",
    "a[3]": "$a_3$",
    "r0[0]": "r₀,₀",
    "r0[1]": "r₀,₁",
    "r0[2]": "r₀,₂",
    "r0[3]": "r₀,₃",
}

def _betterLabels(x):
    """
    Replace strings with their corresponding (greek) symbols
    """
    if isinstance(x, str):
        return _table.get(x,x)
    else:
        return [_table.get(x_,x_) for x_ in x]

def drawPosteriorSamples(trace, nDraws=100, r=np.linspace(2, 8, num=200), t=None):
    VarNames = trace.varnames

    # Determine if a Gaussian model was used and how many iterations were run -------
    GaussianModel = "r0" in VarNames
    if GaussianModel:
        nGaussians = trace['r0'].shape[1]
    
    nChainSamples = trace['P'].shape[0]

    # Generate random indices from chain samples ------------------------------------
    idxSamples = random.sample(range(nChainSamples), nDraws)

    # Draw P's -------------------------------------------------------------------
    Ps = []

    if GaussianModel:
        r0_vecs = trace['r0'][idxSamples]
        w_vecs = trace['w'][idxSamples]
        if nGaussians == 1:
            a_vecs = np.ones_like(idxSamples)
        else:
            a_vecs = trace['a'][idxSamples]

        for iDraw in range(nDraws):
            P = dd_gauss(r,r0_vecs[iDraw],w_vecs[iDraw],a_vecs[iDraw])
            Ps.append(P)
    else:
        for iDraw in range(nDraws):
            P = trace['P'][idxSamples[iDraw]]
            Ps.append(P)

    # Draw corresponding time-domain parameters ---------------------------------
    if 'V0' in VarNames:
        V0 = trace['V0'][idxSamples]

    if 'k' in VarNames:
        k = trace['k'][idxSamples]

    if 'lamb' in VarNames:
        lamb = trace['lamb'][idxSamples]

    # Generate V's from P's and other parameters --------------------------------
    Vs = []
    Bs = []
    K0 = dl.dipolarkernel(t, r, integralop=False)
    dr = r[1] - r[0]

    for iDraw in range(nDraws):
        K_ = copy.copy(K0)
        V_ = dr*K0@Ps[iDraw]

        if 'lamb' in VarNames:
            V_ = (1-lamb[iDraw]) + lamb[iDraw]*V_

        if 'k' in VarNames:
            B = bg_exp(t,k[iDraw])
            V_ *= B
            
            Blamb = (1-lamb[iDraw])*B
            if 'V0' in VarNames:
                Blamb *= V0[iDraw]
            Bs.append(Blamb)

        if 'V0' in VarNames:
            V_ *= V0[iDraw]

        Vs.append(V_)

    return Ps, Vs, Bs, t, r


def plotMCMC(Ps, Vs, Bs, Vdata, t, r, Pref=None, rref=None, show_ave = None):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(11)

    if min(Vs[0])<0.2:
        residuals_offset = -max(Vs[0])/3
    else:
        residuals_offset = 0

    # Plot time-domain quantities
    for V, B in zip(Vs, Bs):
        residuals = V - Vdata
        ax1.plot(t, V, color='#3F60AE', alpha=0.2)
        ax1.plot(t, B, color='#FCC43F', alpha=0.2)
        ax1.plot(t, residuals+residuals_offset, color='#3F60AE', alpha=0.2)
    Vave = np.mean(Vs,0)
    Bave = np.mean(Bs,0)
    Pave = np.mean(Ps,0)
    

    

    

    ax1.scatter(t, Vdata, color='#BFBFBF', s = 5)
    ax1.hlines(residuals_offset, min(t), max(t), color='black')
    ax1.set_xlabel('$t$ (µs)')
    ax1.set_ylabel('$V$ (arb.u.)')
    ax1.set_xlim((min(t), max(t)))
    ax1.set_ylim(-0.1,1.2)
    ax1.set_title('time domain and residuals')

    if show_ave is not None:
        ax1.plot(t,Vave,color='yellow',label= 'Vexp Average')
        ax1.plot(t,Bave,color = 'purple',label = 'Background Average')
    #ax1.plot(t,Vave-residuals,color = 'red')
        

    # Plot distance distributions
    for P in Ps:
        ax2.plot(r, P, color='#3F60AE', alpha=0.2)
    ax2.set_xlabel('$r$ (nm)')
    ax2.set_ylabel('$P$ (nm$^{-1}$)')
    ax2.set_xlim(min(r), max(r))
    ax2.set_ylim(0,2)
    ax2.set_title('distance domain')

    if Pref is not None:
        ax2.plot(rref, Pref, color='black')
    if show_ave is not None: 
        ax2.plot(r,Pave,color = 'black',label = 'Average')

    plt.grid()
    
    return fig



