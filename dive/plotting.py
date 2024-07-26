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
    desiredVars = ["lamb","V0","sigma"]
    if "Bend" in trace.posterior:
        desiredVars.append("Bend")
    elif "tauB" in trace.posterior:
        desiredVars.append("tauB")
    else:
        desiredVars.append("k")
    if "lg_alpha" in trace.posterior:
        desiredVars.append("lg_alpha")

    # checks if gaussian
    if "w_dim_0" in trace.posterior:
        # adds r0, w, and a to the dictionary with the number of variables
        nGauss = trace.posterior.dims["w_dim_0"]
        desiredVars.extend(["r0","w"])
        if nGauss > 1:
            desiredVars.append("a")

    return desiredVars


def printsummary(trace,var_names=None):
    """
    Print table of all parameters, including their means, standard deviations,
    effective sample sizes, Monte Carlo standard errors, and R-hat diagnostics.
    """
    if var_names is None:
        var_names = _relevantVariables(trace)
    summary = az.summary(trace, var_names=var_names)
    # replace the labels with their unicode characters before displaying
    summary.index = _betterLabels(summary.index.values)
    display(summary)

def plotmarginals(trace, axs=None, var_names=None, GroundTruth=None, point_estimate=None, hdi_prob="hide", **kwargs):
    """
    Plot marginalized posteriors
    """
    if var_names is None:
        var_names = _relevantVariables(trace)
    if axs is None:
        fig, axs = plt.subplots(1,len(var_names),figsize=(2.5*len(var_names),2.5))
    for i,ax in enumerate(axs):
        az.plot_posterior(trace, ax=ax, var_names=var_names[i], point_estimate=point_estimate, hdi_prob=hdi_prob, **kwargs)
        ax.patch.set_linewidth(1)
        ax.patch.set_edgecolor("black")
        ax.set_title(_betterLabels(var_names[i]))
        # plots vertical bar @ ground truth if given
        if GroundTruth:
            if var_names[i] in GroundTruth:
                ax.axvline(GroundTruth[var_names[i]], color='black') 
    return axs

def plotcorrelations(trace, axs=None, var_names=None, marginals=True, **kwargs):
    """
    Matrix of pairwise correlation plots between model parameters.
    """
    # determine variables to include
    if var_names is None:
        var_names = _relevantVariables(trace)
    nVars = len(var_names) - 1 + marginals
    
    # configure axes
    if axs is None:
        if nVars < 3:
            figsize = (7, 7)
        else:
            figsize = (10, 10)
        fig, axs = plt.subplots(nVars,nVars,figsize=figsize,layout="constrained")

    # use arviz library to plot correlations
    az.rcParams["plot.max_subplots"] = 200

    az.plot_pair(trace, ax=axs, var_names=var_names, kind='kde', figsize=figsize, marginals=marginals, **kwargs)

    # replace labels with the nicer unicode character versions
    if len(var_names) > 2:
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

    return axs

def plotV(trace, ax=None, nDraws=100, show_avg=False, ci=None, rng=0, residuals_offset=0, Vkwargs={}, Bkwargs={}, reskwargs={}, **kwargs):
    if not ax:
        # creates ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5,5))
    # setup default colors and alphas
    default_colors = ["#4A5899","#F38D68","#4A5899"]
    for i,kwarglist in enumerate([Vkwargs,Bkwargs,reskwargs]):
        if "color" not in kwarglist and "color" not in kwargs:
            kwarglist.update({"color":default_colors[i]})
        if "alpha" not in kwarglist and "alpha" not in kwargs:
            if ci is None:
                kwarglist.update({"alpha":0.2})
            else:
                kwarglist.update({"alpha":0.7})
    # get Vs and Bs from trace
    totalDraws = trace.posterior.dims["chain"]*trace.posterior.dims["draw"]
    Vs, Bs = drawPosteriorSamples(trace, nDraws=(nDraws if ci is None else totalDraws), rng=rng)
    # get t and Vexp from trace
    t = trace.observed_data.coords["V_dim_0"].values
    Vexp = trace.observed_data["V"].values
    # Plot time-domain quantities
    if ci is None:
        for B in Bs:
            ax.plot(t, B, **Bkwargs, **kwargs)
        for V in Vs:
            ax.plot(t, V, **Vkwargs, **kwargs)
            residuals = V - Vexp
            ax.plot(t, residuals+residuals_offset, **reskwargs, **kwargs)
    else:
        # this code may not work in the future
        Bci = az.hdi(np.asarray(Bs),hdi_prob=0.95).transpose()
        ax.fill_between(t, Bci[0], Bci[1], lw=0, **Bkwargs, **kwargs)
        Vci = az.hdi(np.asarray(Vs),hdi_prob=0.95).transpose()
        ax.fill_between(t, Vci[0], Vci[1], lw=0, **Vkwargs, **kwargs)
    # plot average values
    if show_avg:
        Bavg = np.mean(Bs,0)
        ax.plot(t,Bavg,color="black",alpha=0.7,lw=2)
        Vavg = np.mean(Vs,0)
        ax.plot(t,Vavg,color="black",alpha=0.7,lw=2)
    # axis configurations
    ax.plot(t, Vexp, marker=".", color='k', ms=3, alpha=0.6, mew=0, lw=0)
    if ci is None:
        ax.axhline(residuals_offset, color="black")
    ax.set_xlabel('$t$ (µs)')
    ax.set_ylabel('$V(t)$ (arb. u.)')
    ax.set_xlim((min(t), max(t)))
    ax.set_ylim(-0.1+residuals_offset,1.1)
    ax.set_title('time domain and residuals')

def plotP(trace, ax=None, nDraws=100, show_avg=False, ci=None, rng=0, Pref=None, rref=None, alpha=0.2, color="#4A5899", **kwargs):
    if not ax:
        # creates ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5,5))
    # get Ps and r from trace
    totalDraws = trace.posterior.dims["chain"]*trace.posterior.dims["draw"]
    Ps = az.extract(trace, var_names=["P"], num_samples=nDraws, rng=rng).transpose("sample", ...)
    r = trace.posterior.coords["P_dim_0"]
    # Plot distance distributions
    Pmax = 0
    if ci is None:
        for P in Ps:
            ax.plot(r, P, alpha=alpha, color=color, **kwargs)
            if max(P) > Pmax:
                Pmax = max(P)
    else:
        # this may break in the future
        Pci = az.hdi(np.asarray(Ps),hdi_prob=ci).transpose()
        plt.fill_between(r,Pci[0],Pci[1],alpha=alpha,color=color,lw=0,**kwargs)
        Pmax = max(Pci[1])
    # Plot average
    if show_avg:
        Pavg = np.mean(Ps,0)
        ax.plot(r, Pavg, color="black", alpha=0.7, lw=2)
    # axis configurations
    ax.set_xlabel('$r$ (nm)')
    ax.set_ylabel('$P(r)$ (nm$^{-1}$)')
    ax.set_xlim(min(r), max(r))
    ax.set_ylim(0,Pmax*1.1)
    ax.set_title('distance domain')
    if Pref is not None:
        ax.plot(rref, Pref, color='black')
    plt.grid()
    return ax

def summary(trace, var_names=None, nDraws=100, rng=0):
    
    printsummary(trace, var_names)
    plotmarginals(trace, var_names)
    plotcorrelations(trace, var_names)
    fig, axs = plt.subplots(1,2,figsize=(8,4),layout="constrained")
    plotV(trace, ax=axs[0], nDraws=nDraws, rng=rng)
    plotP(trace, ax=axs[1], nDraws=nDraws, rng=rng)

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


def drawPosteriorSamples(trace, nDraws=100, rng=0):
    # Extracts (nDraws) random samples from the trace and reshapes it to work nicely
    varDict = az.extract(trace, num_samples=nDraws, rng=rng).transpose("sample", ...)
    # Extract t and r from trace object
    t = trace.observed_data.coords["V_dim_0"].values
    r = trace.posterior.coords["P_dim_0"].values

    # Rename time-domain parameters to make code below cleaner -------------------------
    if 'P' in varDict:
        Ps = varDict["P"].values
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

    return Vs, Bs

def pairplot_chain(trace, var1, var2, plot_inits=False, gauss_id=1, ax=None, colors=["r","g","b","y","m","c","orange","deeppink","indigo","seagreen"], alpha_points=0.1, alpha_inits=1):
    """Plots two parameters against each other for each chain."""
    if not ax:
        # creates ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5,5))
    gauss_id -= 1 # to fix off-by-one error
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

def pairplot_divergence(trace, var1, var2, gauss_id=1, ax=None, color="C2", divergence_color="C3", alpha=0.2, divergence_alpha=0.4):
    """Plots two parameters against each other and highlights divergences."""
    gauss_id -= 1 # to fix off-by-one error
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

def pairplot_condition(trace, var1, var2, gauss_id=1, ax=None, criterion=None, threshold=None, color_greater="dodgerblue", color_lesser="hotpink", alpha_greater=0.2, alpha_lesser=0.2):
    """Plots two parameters against each other and divides points greater and less than a threshold in a certain criterion."""
    # the criterion should be in sample_stats, e.g. tree_depth
    # points above and below the threshold in this criterion will be plotted in different colors
    gauss_id -= 1 # to fix off-by-one error
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

def plot_hist(trace, var, combine_multi=False, gauss_id=1, ax=None, bins=10, color="k", alpha=1):
    """Plots a histogram of a parameter"""
    if not ax:
        # creates an ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    gauss_id -= 1 # to fix off-by-one error
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