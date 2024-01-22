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


def plotresult(trace, model_dic, nDraws=100, rng=0, Pid=None, Pref=None, rref=None, show_ave=None, chains=None, colors=["#4A5899","#F38D68"]):
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
    
    Ps, Vs, Bs, _, _ = drawPosteriorSamples(trace, nDraws, r, t, rng)
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


def drawPosteriorSamples(trace, nDraws=100, r=np.linspace(2, 8, num=200), t=None, rng=0):
    # Extracts (nDraws) random samples from the trace and reshapes it to work nicely
    varDict = az.extract(trace, num_samples=nDraws, rng=rng).transpose("sample", ...)

    # Draw P's -------------------------------------------------------------------
    Ps = []
    # Determine if a Gaussian model was used
    GaussianModel = "r0" in varDict
    if GaussianModel:
        # if gaussian, build P from r0 (mean), w (width), and a (amplitude)
        r0_vecs = varDict["r0"].values
        w_vecs = varDict["w"].values
        if "a" in varDict:
            a_vecs = varDict["a"].values
        else:
            a_vecs = np.ones(nDraws)
        for iDraw in range(nDraws):
            P = dd_gauss(r,r0_vecs[iDraw],w_vecs[iDraw],a_vecs[iDraw])
            Ps.append(P)
    else:
        # if regularization, simply take P from model
        for iDraw in range(nDraws):
            P = varDict["P"][iDraw]
            Ps.append(P)

    # Rename time-domain parameters to make code below cleaner -------------------------
    if 'V0' in varDict:
        V0 = varDict["V0"].values
    if 'k' in varDict:
        k = varDict["k"].values
    if 'Bend' in varDict:
        Bend = varDict["Bend"].values   
    if 'tauB' in varDict:
        tauB = varDict["tauB"].values
    if 'lamb' in varDict:
        lamb = varDict["lamb"].values

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

def pairplot_chain(trace, var1, var2, plot_inits=False, gauss_id=0, ax=None, colors=["r","g","b","y","m","c","orange","deeppink","indigo","seagreen"], alpha_points=0.1, alpha_inits=1):
    """Plots two parameters against each other for each chain."""
    if not ax:
        # creates ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5,5))
    xlabel = var1
    ylabel = var2
    for chain in range(trace.posterior.dims["chain"]):
        v1 = az.extract(trace, var_names=[var1], combined=False)[chain].transpose()
        v2 = az.extract(trace, var_names=[var2], combined=False)[chain].transpose()
        # for parameters w/ multiple values (a, w, and r0)
        # will plot the values corresponding to gauss_id
        if len(v1.dims) > 1:
                v1 = v1[gauss_id]
                xlabel = var1 + "[%s]" % gauss_id
        if len(v2.dims) > 1:
                v2 = v2[gauss_id]
                ylabel = var2 + "[%s]" % gauss_id
        # if color is a string, it uses that color; if it is a list, it uses them in order
        color = colors if isinstance(colors, str) else colors[chain%len(colors)]
        # plots the two parameters
        ax.plot(v1, v2, ".", color=color, alpha=alpha_points)
        if plot_inits:
            # if plot_inits is True, it plots the initial points as a larger dot
            ax.plot(v1[0], v2[0], "o", color=color, alpha=alpha_inits)
    # labels axes and title
    ax.set_xlabel(_betterLabels(xlabel))
    ax.set_ylabel(_betterLabels(ylabel))
    ax.set_title("scatter plot between %s and %s" % (_betterLabels(xlabel), _betterLabels(ylabel)))
    return ax

def pairplot_divergence(trace, var1, var2, gauss_id=0, ax=None, color="C2", divergence_color="C3", alpha=0.2, divergence_alpha=0.4):
    """Plots two parameters against each other and highlights divergences."""
    v1 = az.extract(trace, var_names=[var1])
    v2 = az.extract(trace, var_names=[var2])
    xlabel = var1
    ylabel = var2
    # for parameters w/ multiple values (a, w, and r0)
    # will plot the values corresponding to gauss_id
    if len(v1.dims) > 1:
        v1 = v1[gauss_id]
        xlabel = var1 + "[%s]" % gauss_id
    if len(v2.dims) > 1:
        v2 = v2[gauss_id]
        ylabel = var2 + "[%s]" % gauss_id
    if not ax:
        # creates an ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    # plots all the points first
    ax.plot(v1, v2, ".", color=color, alpha=alpha)
    # then, plots the divergent points in divergence_color & larger
    divergent = az.extract(trace, group="sample_stats", var_names=["diverging"])
    ax.plot(v1[divergent], v2[divergent], "o", color=divergence_color, alpha=divergence_alpha)
    ax.set_xlabel(_betterLabels(xlabel))
    ax.set_ylabel(_betterLabels(ylabel))
    ax.set_title("scatter plot with divergences between %s and %s" % (_betterLabels(xlabel), _betterLabels(ylabel)))
    return ax

def pairplot_condition(trace, var1, var2, gauss_id=0, ax=None, criterion=None, threshold=None, color_greater="dodgerblue", color_lesser="hotpink", alpha_greater=0.2, alpha_lesser=0.2):
    """Plots two parameters against each other and divides points greater and less than a threshold in a certain criterion."""
    # the criterion should be in sample_stats, e.g. tree_depth
    # points above and below the threshold in this criterion will be plotted in different colors
    v1 = az.extract(trace, var_names=[var1])
    v2 = az.extract(trace, var_names=[var2])
    stats = az.extract(trace, group="sample_stats", var_names=[criterion])
    xlabel = var1
    ylabel = var2
    # for parameters w/ multiple values (a, w, and r0)
    # will plot the values corresponding to gauss_id
    if len(v1.dims) > 1:
        v1 = v1[gauss_id]
        xlabel = var1 + "[%s]" % gauss_id
    if len(v2.dims) > 1:
        v2 = v2[gauss_id]
        ylabel = var2 + "[%s]" % gauss_id
    if not ax:
        # creates an ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    for i in range(len(v1)):
        if stats[i] > threshold:
            ax.plot(v1[i], v2[i], ".", color=color_greater, alpha=alpha_greater)
        else:
            ax.plot(v1[i], v2[i], ".", color=color_lesser, alpha=alpha_lesser)
    ax.set_xlabel(_betterLabels(xlabel))
    ax.set_ylabel(_betterLabels(ylabel))
    ax.set_title("scatter plot between %s and %s split at %s = %s" % (_betterLabels(xlabel), _betterLabels(ylabel), criterion, threshold))
    return ax

def plot_hist(trace, var, combine_multi=False, gauss_id=0, ax=None, bins=10, color="k", alpha=1):
    """Plots a histogram of a parameter"""
    if not ax:
        # creates an ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    xlabel = var
    v = az.extract(trace, var_names=[var])
    if len(v.dims) > 1:
        if combine_multi: # for multi-value parameters a, w, and r0, will combine into one histogram
            v = v.unstack().stack(stacked=["draw",...]) # combines into one list
        else:
            v = v[gauss_id] # selects the gaussian chosen in gauss_id
            xlabel = var + "[%s]" % gauss_id
    ax.set_xlabel(_betterLabels(xlabel))
    ax.set_ylabel("number of draws")
    ax.set_title("histogram of %s" % _betterLabels(xlabel))
    ax.hist(v, bins=bins, alpha=alpha, color=color)