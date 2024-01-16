## Plotting 

# Import modules
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
import numpy as np
import random
import arviz as az
from IPython.display import display
import deerlab as dl
from scipy.io import loadmat

from .utils import *
from .deer import *


def _relevantVariables(trace):
    #desiredVars = ["r0", "w", "a", "k", "lamb", "V0", "sigma", "lg_alpha"]
    if "Bend" in trace.posterior:
        desiredVars = {"Bend": 1, "lamb": 1, "V0": 1, "sigma": 1}
    elif "tauB" in trace.posterior:
        desiredVars = {"tauB": 1, "lamb": 1, "V0": 1, "sigma": 1}
    else:
        desiredVars = {   "k": 1, "lamb": 1, "V0": 1, "sigma": 1}
    if "lg_alpha" in trace.posterior:
        desiredVars.update({"lg_alpha": 1})
    # creates a dictionary with the variables and how many of them there are (None for single ones)
    Vars = {key: desiredVars[key] for key in desiredVars if key in trace.posterior}

    # checks if gaussian
    if "w_dim_0" in trace.posterior:
        # adds r0, w, and a to the dictionary with the number of variables
        nGauss = trace.posterior.dims["w_dim_0"]
        Vars.update({"r0": nGauss, "w": nGauss})
        if nGauss > 1:
            Vars.update({"a": nGauss})

    return Vars


def printsummary(trace, model_dic):
    """
    Print table of all parameters, including their means, standard deviations,
    effective sample sizes, Monte Carlo standard errors, and R-hat diagnostics.
    """
    Vars = _relevantVariables(trace)
    with model_dic['model']:
        summary = az.summary(trace, var_names=list(Vars.keys()))
    # replace the labels with their unicode characters before displaying
    summary.index = _betterLabels(summary.index.values)
    display(summary)


def plotmarginals(trace, GroundTruth=None, nCols=6):
    """
    Plot marginalized posteriors
    """
    Vars = _relevantVariables(trace)
    nVars = sum(Vars.values())

    # figure out layout of plots and create figure
    nCols = min(nVars,nCols)
    nRows = int(np.ceil(nVars/nCols))
    fig, axs = plt.subplots(nRows, nCols)
    axs = axs.flatten()
    width = min(3*nVars,12)
    height = nRows*3.5

    # set figure size
    fig.set_figheight(height)
    fig.set_figwidth(width)
    
    # KDE of chain samples and plot them
    ax_id=0
    for key in Vars:
        if Vars[key] == 1:
            if key in trace.posterior:
                az.plot_kde(np.array([np.ndarray.item(draw.values) for chain in trace.posterior[key] for draw in chain]), ax=axs[ax_id])

            axs[ax_id].set_xlabel(_betterLabels(key), fontsize='large')
            axs[ax_id].yaxis.set_ticks([])
            axs[ax_id].grid(axis='x')

            if GroundTruth:
                if key in GroundTruth.keys():
                    bottom, top = axs[ax_id].get_ylim()
                    axs[ax_id].vlines(GroundTruth[key], bottom, top, color='black')
            ax_id += 1
        else:
            for i in range(Vars[key]):
                if key in trace.posterior:
                    az.plot_kde(np.array([np.ndarray.item(draw.values) for chain in trace.posterior[key][i] for draw in chain]), ax=axs[ax_id])
                axs[ax_id].set_xlabel(_betterLabels(key+"[%s]"%i), fontsize='large')
                axs[ax_id].yaxis.set_ticks([])
                axs[ax_id].grid(axis='x')

                if GroundTruth:
                    if key in GroundTruth.keys():
                        bottom, top = axs[ax_id].get_ylim()
                        axs[ax_id].vlines(GroundTruth[key], bottom, top, color='black')
                ax_id += 1

    # Clean up figure
    for j in range(nVars, len(axs)):
        axs[j].axis('off')
    fig.tight_layout()
    return fig


def plotcorrelations(trace, model_dic, figsize=None, marginals=True, div=False):
    """
    Matrix of pairwise correlation plots between model parameters.
    """
    # determine variables to include
    Vars = _relevantVariables(trace)
    nVars = sum(Vars.values())
    
    # Set default figure size
    if figsize is None:
        if nVars < 3:
            figsize = (7, 7)
        else:
            figsize = (10, 10)
    if div == True:
        class Object(object):
            pass

        trace.sample_stats = Object()
        trace.sample_stats.diverging = trace.diverging
    # use arviz library to plot correlations
    az.rcParams["plot.max_subplots"] = 200
    with model_dic["model"]:
        axs = az.plot_pair(trace, var_names=list(Vars.keys()), kind='kde', figsize=figsize, marginals=marginals, divergences=div)

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


def plotresult(trace, model_dic, nDraws=100, Pid=None, Pref=None, rref=None, show_ave=None, chains=None, colors=["#4A5899","#F38D68"]):
    """
    Plot the MCMC results in the time domain and in the distance domain, using an
    ensemble of P draws from the posterior, and the associated time-domain signals.
    Also shown in the time domain: the ensemble of residual vectors, and the ensemble
    of backgrounds.
    """
#displaying posterior average data:
    if show_ave is not None:
        print('Showing posterior average')
    
    fig1 = []
    if chains is not None:
       fig1 =az.plot_trace(trace)
        


    
    fig1 = []
    if chains is not None:
       fig1 =az.plot_trace(trace)
        

    # Get reference distribution if specified ------------------------------------
    if Pid is not None:
        refdata = loadmat('data/edwards_testset/distributions_2LZM.mat')
        P0s = refdata['P0']
        rref = np.squeeze(refdata['r0'])
        Pref = P0s[Pid-1,:]
    
    elif Pref is not None:
        if rref is None:
            raise KeyError("If 'Pref' is provided, 'rref' must be provided as well.")

    Vexp = model_dic['Vexp']
    t = model_dic['t']
    r = model_dic['pars']['r']
    
    Ps, Vs, Bs, _, _ = drawPosteriorSamples(trace, nDraws, r, t)
    fig = plotMCMC(Ps, Vs, Bs, Vexp, t, r, Pref, rref, show_ave, colors)

    return fig, fig1

# Look-up table that maps variable strings to better symbols for printing
_table = {
    "k": "$k$",
    "tau": "$τ_B$",
    "Bend": "$B_\mathrm{end}$",
    "lamb": "$λ$",
    "lamba": "$λ$",
    "sigma": "$σ$",
    "delta": "$δ$",
    "tau": "$τ$",
    "V0": "$V_0$",
    "r0": "$r_0$",
    "alpha": "$α$",
    "lg_alpha": "$\mathrm{lg}(α)$",
    "w\n0": "$w_1$",
    "w\n1": "$w_2$",
    "w\n2": "$w_3$",
    "w\n3": "$w_4$",
    "w[0]": "$w_1$",
    "w[1]": "$w_2$",
    "w[2]": "$w_3$",
    "w[3]": "$w_4$",
    "a[0]": "$a_1$",
    "a[1]": "$a_2$",
    "a[2]": "$a_3$",
    "a[3]": "$a_4$",
    "a\n0": "$a_1$",
    "a\n1": "$a_2$",
    "a\n2": "$a_3$",
    "a\n3": "$a_4$",
    "r0[0]": "$r_{0,1}$",
    "r0[1]": "$r_{0,2}$",
    "r0[2]": "$r_{0,3}$",
    "r0[3]": "$r_{0,4}$",
    "r0\n0": "$r_{0,1}$",
    "r0\n1": "$r_{0,2}$",
    "r0\n2": "$r_{0,3}$",
    "r0\n3": "$r_{0,4}$",
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

    varDict = {key: [draw.values for chain in trace.posterior[key] for draw in chain] for key in trace.posterior}

    # Determine if a Gaussian model was used and how many iterations were run -------
    GaussianModel = "r0" in varDict
    if GaussianModel:
        nGaussians = len(varDict['r0'][0])

    nChainSamples = len(varDict['P'])

    # Generate random indices for chain samples ------------------------------------
    idxSamples = random.sample(range(nChainSamples), nDraws)

    # Draw P's -------------------------------------------------------------------
    Ps = []

    if GaussianModel:
        r0_vecs = [varDict["r0"][i] for i in idxSamples]
        w_vecs = [varDict["w"][i] for i in idxSamples]
        if nGaussians == 1:
            a_vecs = np.ones_like(idxSamples)
        else:
            a_vecs = [varDict["a"][i] for i in idxSamples]

        for iDraw in range(nDraws):
            P = dd_gauss(r,r0_vecs[iDraw],w_vecs[iDraw],a_vecs[iDraw])
            Ps.append(P)
    else:
        for iDraw in range(nDraws):
            P = varDict["P"][idxSamples[iDraw]]
            Ps.append(P)

    # Draw corresponding time-domain parameters ---------------------------------
    if 'V0' in varDict:
        V0 = [varDict["V0"][i] for i in idxSamples]

    if 'k' in varDict:
        k = [varDict["k"][i] for i in idxSamples]
        
    if 'Bend' in varDict:
        Bend = [varDict["Bend"][i] for i in idxSamples]
        
    if 'tauB' in varDict:
        tauB = [varDict["tauB"][i] for i in idxSamples]

    if 'lamb' in varDict:
        lamb = [varDict["lamb"][i] for i in idxSamples]

    # Generate V's and B's from P's and other parameters --------------------------------
    Vs = []
    Bs = []
    K0 = dl.dipolarkernel(t, r, integralop=False)
    dr = r[1] - r[0]

    for iDraw in range(nDraws):
        V_ = dr*K0@Ps[iDraw]

        if 'lamb' in varDict:
            V_ = (1-lamb[iDraw]) + lamb[iDraw]*V_

        if 'Bend' in varDict:
            k = -1/t[-1]*np.log(Bend[iDraw])
            B = bg_exp(t,k)
        elif 'tauB' in varDict:
            B = bg_exp_time(t,tauB[iDraw])
        else:
            B = bg_exp(t,k[iDraw])
        
        V_ *= B
        Blamb = (1-lamb[iDraw])*B
            
        if 'V0' in varDict:
            Blamb *= V0[iDraw]
        Bs.append(Blamb)
        
        if 'tauB' in varDict:
            B = bg_exp_time(t,tauB[iDraw])
            V_ *= B

            Blamb = (1-lamb[iDraw])*B
            if 'V0' in varDict:
                Blamb *= V0[iDraw]
            Bs.append(Blamb)

        if 'V0' in varDict:
            V_ *= V0[iDraw]

        Vs.append(V_)

    return Ps, Vs, Bs, t, r


def plotMCMC(Ps, Vs, Bs, Vdata, t, r, Pref=None, rref=None, show_ave = None, colors=["#4A5899","#F38D68"]):



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
        ax1.plot(t, V, color=colors[0], alpha=0.2)
        ax1.plot(t, B, color=colors[1], alpha=0.2)
        ax1.plot(t, residuals+residuals_offset, color=colors[0], alpha=0.2)
    Vavg = np.mean(Vs, 0)
    Bavg = np.mean(Bs, 0)
    Pavg = np.mean(Ps, 0)

    ax1.scatter(t, Vdata, color='#BFBFBF', s=5)
    ax1.hlines(residuals_offset, min(t), max(t), color='black')
    ax1.set_xlabel('$t$ (µs)')
    ax1.set_ylabel('$V$ (arb.u.)')
    ax1.set_xlim((min(t), max(t)))
    ax1.set_ylim(-0.1,1.1)
    ax1.set_title('time domain and residuals')


    if show_ave is not None:
        ax1.plot(t,Vave,color='yellow',label= 'Vexp Average')
        ax1.plot(t,Bave,color = 'purple',label = 'Background Average')
    #ax1.plot(t,Vave-residuals,color = 'red')
        


    # Plot distance distributions
    for P in Ps:
        ax2.plot(r, P, color=colors[0], alpha=0.2)
    Pmax = max([max(P) for P in Ps])
    ax2.set_xlabel('$r$ (nm)')
    ax2.set_ylabel('$P$ (nm$^{-1}$)')
    ax2.set_xlim(min(r), max(r))
    ax2.set_ylim(0,max(P)+0.2)
    ax2.set_title('distance domain')

    if Pref is not None:
        ax2.plot(rref, Pref, color='black')
    if show_ave is not None: 
        ax2.plot(r,Pave,color = 'black',label = 'Average')

    plt.grid()
        
    return fig

def pairplot_chain(trace, var1, var2, plot_inits=True, ax=None, colors=["r","g","b","y","m","c","orange","deeppink","indigo","seagreen"], alpha=0.2, alpha_inits=1):
    for chain in range(trace.posterior.dims["chain"]):
        v1 = np.array([draw.values for draw in trace.posterior[var1][chain]]).flatten()
        v2 = np.array([draw.values for draw in trace.posterior[var2][chain]]).flatten()
        if not ax:
            _, ax = plt.subplots(1, 1, figsize=(5,5))
        color = colors if isinstance(colors, str) else (colors[chain] if chain < len(colors) else colors[chain%len(colors)])
        ax.plot(v1, v2, ".", color=color, alpha=alpha)
        ax.set_xlabel(_betterLabels(var1))
        if plot_inits:
            ax.plot(v1[0], v2[0], "o", color=color, alpha=alpha_inits)
        ax.set_ylabel(_betterLabels(var2))
        ax.set_title("scatter plot between %s and %s" % (_betterLabels(var1), _betterLabels(var2)))
    return ax

def pairplot_divergence(trace, var1, var2, ax=None, color="C2", divergence_color="C3", alpha=0.2, divergence_alpha=0.4):
    v1 = np.array([draw.values for chain in trace.posterior[var1] for draw in chain]).flatten()
    v2 = np.array([draw.values for chain in trace.posterior[var2] for draw in chain]).flatten()
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(v1, v2, ".", color=color, alpha=alpha)
    divergent = np.array([draw.values for chain in trace.sample_stats.diverging for draw in chain]).flatten()
    ax.plot(v1[divergent], v2[divergent], "o", color=divergence_color, alpha=divergence_alpha)
    ax.set_xlabel(_betterLabels(var1))
    ax.set_ylabel(_betterLabels(var2))
    ax.set_title("scatter plot with divergences between %s and %s" % (_betterLabels(var1), _betterLabels(var2)))
    return ax