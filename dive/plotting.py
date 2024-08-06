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
from xarray.core.utils import HiddenKeyDict

from .utils import *
from .deer import *

from typing import Union
from numpy.typing import ArrayLike
from matplotlib.typing import ColorType

def _get_relevant_vars(trace: az.InferenceData) -> list[str]:
    """Returns a list of important variables from a given trace.

    These variables are used for some plotting functions.

    The important variables are:
        lamb        modulation depth
        V0          signal amplitude
        Bend        value at end of background function
        lg_alpha    log10 of regularizaton parameter alpha
        r0          mean(s) of gaussian(s) of distance distribution
        w           width(s) of gaussian(s) of distance distribution
        a           amplitudes of gaussians of distance distribution
    
    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.

    Returns
    -------
    relevant_vars : list of str
        A list of variables present in the trace that are important
        to plot or summarize.
    """
    # list of important variables possibly in the trace
    possible_vars = ["lamb","V0","sigma","lg_alpha","r0","w","a","Bend"]
    relevant_vars = []
    for var in possible_vars:
        if var in trace.posterior:
            relevant_vars.append(var)
    # only add k if Bend not present
    if "k" in trace.posterior and "Bend" not in relevant_vars:
        relevant_vars.append("k")
    return relevant_vars

def print_summary(
    trace: az.InferenceData, var_names: list[str] = None, **kwargs):
    """Prints a table with summary statistics of important parameters.
    
    The table includes their means, standard deviations, 
    effective sample sizes, Monte Carlo standard errors, and R-hat 
    diagnostic values.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    var_names : list of str, optional
        The variables to be summarized. If not provided, a list of
        relevant important variables will be automatically selected.
    **kwargs : dict, optional
        Keyword arguments to be passed to az.summary.

    See Also
    --------
    az.summary
    summary
    """
    # get relevant variables if not provided
    if var_names is None:
        var_names = _get_relevant_vars(trace)
    summary = az.summary(trace, var_names=var_names, **kwargs)
    # replace the labels with their unicode characters before displaying
    summary.index = _replace_labels(summary.index.values)
    display(summary)
    return

def plot_marginals(
    trace: az.InferenceData, axs: np.ndarray[plt.Axes] = None, 
    var_names: list[str] = None, ground_truth: dict[str,float] = None, 
    point_estimate: str = None, hdi_prob: float = "hide", **kwargs) -> plt.Axes:
    """Plot 1D marginalized posteriors of selected variables.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    axs : np.ndarray of plt.Axes, optional
        1D numpy array of the MatPlotLib axes to be plotted on. If not 
        provided, axes will be automatically created. Needs to be the 
        correct size.
    var_names : list of str, optional
        The variables to be plotted. If not provided, a list of
        relevant important variables will be automatically selected.
    ground_truth : dict of str, float, optional
        A dictionary of ground-truth variable values, which will be
        plotted as vertical gray lines. Keys should be the variable
        names and arguments should be their values.
    point_estimate : str, optional
        If "mean", "median", or "mode" are passed, that value will
        be plotted. See az.plot_posterior.
    hdi_prob : float, default="hide"
        The highest density interval to plot. If set to the default
        "hide", none will be plotted.
    **kwargs : dict, optional
        Keyword arguments to be passed to az.plot_posterior.
    
    Returns
    -------
    axs : plt.Axes
    
    See Also
    --------
    az.plot_posterior
    summary
    """
    # get relevant variables if not provided
    if var_names is None:
        var_names = _get_relevant_vars(trace)
    # create axes if not provided
    if axs is None:
        figsize = (2.5*len(var_names),2.5)
        fig, axs = plt.subplots(1,len(var_names),figsize=figsize)
    # plots and stylizes posterior for each variable
    for i,ax in enumerate(axs):
        az.plot_posterior(trace, ax=ax, var_names=var_names[i], 
                          point_estimate=point_estimate, hdi_prob=hdi_prob, 
                          **kwargs)
        ax.patch.set_linewidth(1)
        ax.patch.set_edgecolor("black")
        ax.set_title(_replace_labels(var_names[i]))
        # plots vertical bar @ ground truth if given
        if ground_truth:
            if var_names[i] in ground_truth:
                ax.axvline(ground_truth[var_names[i]], color='black') 
    return axs

def plot_correlations(
    trace: az.InferenceData, axs: np.ndarray[plt.Axes] = None, 
    var_names: list[str] = None, marginals: bool = True, **kwargs) -> plt.Axes:
    """Plots 2D marginalized posteriors of selected variables.

    Illustrates pairwise correlation plots between model parameters.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    axs : np.ndarray of plt.Axes
        2D numpy array of the MatPlotLib axes to be plotted on. If not
        provided, axes will be automatically created. Needs to be the
        correct size.
    var_names : list of str
        The variables to be plotted. If not provided, a list of relevant
        important variables will be automatically selected.
    marginals : bool, default=True
        Whether or not to also include the 1D marginalized posteriors
        for each variable.
    **kwargs : dict, optional
        Keyword arguments to be passed to az.plot_pair.
    
    Returns
    -------
    axs : plt.Axes

    See Also
    --------
    az.plot_pair
    summary
    """
    # determine variables to include
    if var_names is None:
        var_names = _get_relevant_vars(trace)
    n_vars = len(var_names) - 1 + marginals
    # edit n_vars for gaussians
    for var in ["r0","w","a"]:
        if var in var_names:
            dim_name = var + "_dim_0"
            if dim_name in trace.posterior.sizes:
                n_vars += trace.posterior.sizes[dim_name]-1
    # configure axes
    if axs is None:
        if n_vars < 3:
            figsize = (7, 7)
        else:
            figsize = (10, 10)
        fig, axs = plt.subplots(n_vars,n_vars,figsize=figsize,
                                layout="constrained")

    # use arviz library to plot correlations
    az.rcParams["plot.max_subplots"] = 200
    az.plot_pair(trace, ax=axs, var_names=var_names, kind='kde', 
                 figsize=figsize, marginals=marginals, **kwargs)

    # replace labels with the nicer unicode character versions
    if len(var_names) > 2:
        # reshape axes so that we can loop through them
        axs = np.reshape(axs,np.shape(axs)[0]*np.shape(axs)[1])
        for ax in axs:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel:
                ax.set_xlabel(_replace_labels(xlabel))
            if ylabel:
                ax.set_ylabel(_replace_labels(ylabel))
    else:
        xlabel = axs.get_xlabel()
        ylabel = axs.get_ylabel()
        axs.set_xlabel(_replace_labels(xlabel))
        axs.set_ylabel(_replace_labels(ylabel))
    return axs

def plot_V(
    trace: az.InferenceData, ax: plt.Axes = None, num_samples: int = 100, 
    show_avg: bool = False, hdi: float = None, rng: int = None, 
    residuals_offset: float = 0, V_kwargs: dict = {}, B_kwargs: dict = {}, 
    res_kwargs = {}, **kwargs) -> plt.Axes:
    """Plots an ensemble of fitted signals and backgrounds with residuals.

    A set of random samples from the full trace is selected for V and B. 
    Averages and highest density intervals may also be optionally 
    plotted.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    ax : plt.Axes, optional
        The MatPlotLib axes to plot on. If none provided, axes will be
        automatically created. 
    num_samples : int, default=100
        The number of random samples to draw from the trace.
    show_avg : bool, default=False
        Whether or not to plot the average signal and background.
    hdi : float, optional
        If a float is provided, the corresponding highest density 
        interval will be plotted instead of the ensembles.
    rng : int, optional
        The seed for the random number generator for drawing random
        samples.
    residuals_offset : float, default=0
        The amount to raise the residual plot by (for a more compact
        plot).
    V_kwargs : dict, default={}
        Keyword arguments to be passed to plt.plot or plt.fill_between 
        for V.
    B_kwargs : dict, default={}
        Keyword arguments to be passed to plt.plot or plt.fill_between 
        for B.
    res_kwargs : dict, default={}
        Keyword arguments to be passed to plt.plot or plt.fill_between 
        for the residuals.
    **kwargs : dict, optional
        Keyword arguments to be passed to plt.plot or plt.fill_between 
        for all plots.
    
    Returns
    -------
    ax : plt.Axes

    See Also
    --------
    draw_posterior_samples
    summary
    plt.plot
    plt.fill_between
    """
    if not ax:
        # creates ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5,5))
    # setup default colors and alphas
    default_colors = ["#4A5899","#F38D68","#4A5899"]
    for i,kwarglist in enumerate([V_kwargs,B_kwargs,res_kwargs]):
        if "color" not in kwarglist and "color" not in kwargs:
            kwarglist.update({"color":default_colors[i]})
        if "alpha" not in kwarglist and "alpha" not in kwargs:
            if hdi is None:
                kwarglist.update({"alpha":0.2})
            else:
                kwarglist.update({"alpha":0.7})
    # get Vs and Bs from draw_posterior_samples
    if hdi is not None:
        num_samples = trace.posterior.sizes["chain"]*trace.posterior.sizes["draw"]
    Vs, Bs = draw_posterior_samples(trace, num_samples, rng=rng)
    # get t and Vexp from trace
    t = trace.observed_data.coords["V_dim_0"].values
    Vexp = trace.observed_data["V"].values
    # Plot time-domain quantities
    if hdi is None:
        for B in Bs:
            ax.plot(t, B, **kwargs, **B_kwargs)
        for V in Vs:
            ax.plot(t, V, **kwargs, **V_kwargs)
            residuals = V - Vexp
            ax.plot(t, residuals+residuals_offset, **kwargs, **res_kwargs)
    else:
        # this code may not work in the future
        Bhdi = az.hdi(np.asarray(Bs),hdi_prob=0.95).transpose()
        ax.fill_between(t, Bhdi[0], Bhdi[1], lw=0, **kwargs, **B_kwargs)
        Vhdi = az.hdi(np.asarray(Vs),hdi_prob=0.95).transpose()
        ax.fill_between(t, Vhdi[0], Vhdi[1], lw=0, **kwargs, **V_kwargs)
    # plot average values
    if show_avg:
        Bavg = np.mean(Bs,0)
        ax.plot(t,Bavg,color="black",alpha=0.7,lw=2)
        Vavg = np.mean(Vs,0)
        ax.plot(t,Vavg,color="black",alpha=0.7,lw=2)
    # axis configurations
    ax.plot(t, Vexp, marker=".", color='k', ms=3, alpha=0.6, mew=0, lw=0)
    if hdi is None:
        ax.axhline(residuals_offset, color="black")
    ax.set_xlabel('$t$ (µs)')
    ax.set_ylabel('$V(t)$ (arb. u.)')
    ax.set_xlim((min(t), max(t)))
    ax.set_ylim(-0.1+residuals_offset,1.1)
    ax.set_title('time domain and residuals')
    return ax

def plot_P(
    trace : az.InferenceData, ax: plt.Axes = None, num_samples: int = 100, 
    show_avg: bool = False, hdi: float = None, rng: int = None, 
    Pref: ArrayLike = None, rref: ArrayLike = None, alpha: float = 0.2, 
    color: ColorType = "#4A5899", **kwargs) -> plt.Axes:
    """Plots an ensemble of distance distributions.

    A set of random distance distributions is drawn from the posterior
    of P. Averages and highest density intervals may also be optionally
    plotted. A ground-truth distance distribution will be plotted in
    black if provided through rref and Pref.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    ax : plt.Axes, optional
        The MatPlotLib axes to plot on. If none provided, axes will be
        automatically created. 
    num_samples : int, default=100
        The number of random samples to draw from the trace.
    show_avg : bool, default=False
        Whether or not to plot the average distance distribution.
    hdi : float, optional
        If a float is provided, the corresponding highest density 
        interval will be plotted instead of the ensembles.
    rng : int, optional
        The seed for the random number generator for drawing random
        samples.
    Pref : ArrayLike, optional
        The ground-truth distance distribution.
    rref : ArrayLike, optional
        The distnace axis of the ground-truth distance distribution.
    alpha : float, default=0.2
        The transparency of the ensemble plots.
    color : ColorType, default="#4A5899"
        The color of the ensemble plots.
    **kwargs : dict, optional
        Keyword arguments to be passed to plt.plot or plt.fill_between.
    
    Returns
    -------
    ax : plt.Axes

    See Also
    --------
    az.extract
    summary
    plt.plot
    plt.fill_between
    """
    if not ax:
        # creates ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5,5))
    # get Ps and r from trace
    if hdi is not None:
        num_samples = trace.posterior.sizes["chain"]*trace.posterior.sizes["draw"]
    Ps = az.extract(trace, var_names=["P"], num_samples=num_samples, 
                    rng=rng).transpose("sample", ...)
    r = trace.posterior.coords["P_dim_0"]
    # Plot distance distributions
    Pmax = 0
    if hdi is None:
        for P in Ps:
            ax.plot(r, P, alpha=alpha, color=color, **kwargs)
            if max(P) > Pmax:
                Pmax = max(P)
    else:
        # this may break in the future
        Pci = az.hdi(np.asarray(Ps),hdi_prob=hdi).transpose()
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
    if Pref is not None and rref is not None:
        ax.plot(rref, Pref, color='black')
    plt.grid()
    return ax

def summary(
    trace: az.InferenceData, var_names: list[str] = None, 
    num_samples: int = 100, rng: int = None):
    """Summary function to plot several plots.

    Calls print_summary, plot_marginals, plot_correlations, plot_V,
    and plot_P.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    var_names : list of str
        The variables to be plotted. If not provided, a list of relevant
        important variables will be automatically selected.
    num_samples : int, default=100
        The number of random samples to draw from the trace.
    rng : int, optional
        The seed for the random number generator for drawing random
        samples.

    See Also
    --------
    print_summary
    plot_marginals
    plot_correlations
    plot_V
    plot_P
    """
    print_summary(trace, var_names)
    plot_marginals(trace, var_names)
    plot_correlations(trace, var_names)
    fig, axs = plt.subplots(1,2,figsize=(8,4),layout="constrained")
    plot_V(trace, ax=axs[0], num_samples=num_samples, rng=rng)
    plot_P(trace, ax=axs[1], num_samples=num_samples, rng=rng)
    return

# Look-up table that maps variable strings to better symbols for printing
_table = {
    "k": "$k$",
    "tauB": "$τ_B$",
    "Bend": "$B_\mathrm{end}$",
    "lamb": "$λ$",
    "lambda": "$λ$",
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


def _replace_labels(x: Union[str, list[str]]) -> Union[str, list[str]]:
    """Replaces strings with their corresponding (greek) symbols.

    Parameters
    ----------
    x : str or list of str
        The label to be replaced.

    Returns
    -------
    str or list of str
        The replaced label.
    """
    if isinstance(x, str):
        return _table.get(x,x)
    else:
        return [_table.get(x_,x_) for x_ in x]


def draw_posterior_samples(
    trace: az.InferenceData, num_samples: int = 100, 
    rng: int = None) -> tuple[np.ndarray,np.ndarray]:
    """Draws random samples from a trace and calculates Vs and Bs.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    num_samples : int, default=100
        The number of random samples to draw from the trace.
    rng : int, optional
        The seed for the random number generator for drawing random
        samples.

    Returns
    -------
    Vs, Bs : tuple of np.ndarray
        The fitted signals and backgrounds from the random samples.

    See Also
    --------
    plot_V
    """
    # Extracts (nDraws) random samples from the trace and reshapes it
    var_dict = az.extract(trace, num_samples=num_samples, 
                          rng=rng).transpose("sample", ...)
    # Extract t and r from trace object
    t = trace.observed_data.coords["V_dim_0"].values
    r = trace.posterior.coords["P_dim_0"].values

    # Rename time-domain parameters to make code below cleaner
    if 'P' in var_dict:
        Ps = var_dict["P"].values
    if 'V0' in var_dict:
        V0 = var_dict["V0"].values
    if 'k' in var_dict:
        k = var_dict["k"].values
    if 'Bend' in var_dict:
        Bend = var_dict["Bend"].values   
    if 'lamb' in var_dict:
        lamb = var_dict["lamb"].values

    # Generate V's and B's from P's and other parameters
    Vs = []
    Bs = []
    K0 = dl.dipolarkernel(t, r, integralop=False)
    dr = r[1] - r[0]

    for i in range(num_samples):
        V_ = dr*K0@Ps[i]
        if 'lamb' in var_dict:
            V_ = (1-lamb[i]) + lamb[i]*V_
        if 'Bend' in var_dict:
            k = -1/t[-1]*np.log(Bend[i])
            B = bg_exp(t,k)
        else:
            B = bg_exp(t,k[i])
        V_ *= B
        Blamb = (1-lamb[i])*B
        if 'V0' in var_dict:
            Blamb *= V0[i]
            V_ *= V0[i]

        Bs.append(Blamb)  
        Vs.append(V_)

    return Vs, Bs

def pairplot_chain(
    trace: az.InferenceData, var1: str, var2: str, ax: plt.Axes = None, 
    plot_inits: bool = False, gauss_id: int = 1, 
    colors: Union[list[ColorType],ColorType] = ["r","g","b","y","m","c"], 
    alpha: float = 0.1, **kwargs) -> plt.Axes:
    """Plots a scatter plot of two parameters for each chain.

    Each chain will be plotted in a different color. Optionally, the
    initial point of each chain may be plotted.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    var1, var2 : str
        The variables to be plotted.
    ax : plt.Axes, optional
        The MatPlotLib axes to plot on. If none provided, axes will be
        automatically created. 
    plot_inits : bool, default=False
        Whether or not to plot the initial points of each chain.
    gauss_id : int, default=1
        For the parameters associated with a gaussian fit (r0, w, a),
        which gaussian to plot. E.g., if var1 is set to w and gauss_id
        is set to 2, the widths of the second gaussian will be plotted.
        Counting starts at 1.
    colors : ColorType or list of ColorType, default=["r","g","b","y","m","c"]
        The color(s) to plot the chains. If a str is provided, all
        chains will be plotted that color. If a list of colors is
        provided, the chains will follow those colors. If the number of
        colors is less than the number of chains, the colors will be
        cycled.
    alpha : float, default=0.1
        The transparency of the plotted points.
    **kwargs : dict, optional
        Keyword arguments to be passed to plt.plot.
    
    Returns
    -------
    ax : plt.Axes

    See Also
    --------
    plt.plot
    """
    if not ax:
        # creates ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5,5))
    gauss_id -= 1 # to fix off-by-one error
    xlabel = var1
    ylabel = var2
    for chain in range(trace.posterior.dims["chain"]):
        # retrieve variable values
        v1 = az.extract(trace, var_names=[var1], 
                        combined=False)[chain].transpose()
        v2 = az.extract(trace, var_names=[var2], 
                        combined=False)[chain].transpose()
        # for parameters w/ multiple values (a, w, and r0)
        # will plot the values corresponding to gauss_id
        if len(v1.dims) > 1:
                v1 = v1[gauss_id]
                xlabel = var1 + "[%s]" % gauss_id
        if len(v2.dims) > 1:
                v2 = v2[gauss_id]
                ylabel = var2 + "[%s]" % gauss_id
        # if color is a string, it uses that color
        # if it is a list, it uses them in order
        color = colors if isinstance(colors, str) else colors[chain%len(colors)]
        # plots the two parameters
        ax.plot(v1, v2, ".", color=color, alpha=alpha, **kwargs)
        if plot_inits:
            # if plot_inits is True, it plots the initial points as a larger dot
            ax.plot(v1[0], v2[0], "o", color=color)
    # labels axes and title
    ax.set_xlabel(_replace_labels(xlabel))
    ax.set_ylabel(_replace_labels(ylabel))
    ax.set_title("scatter plot between %s and %s" 
                 % (_replace_labels(xlabel), _replace_labels(ylabel)))
    return ax

def pairplot_divergence(
    trace: az.InferenceData, var1: str, var2: str, ax: plt.Axes=None, 
    gauss_id: int = 1, color: ColorType = "C2", 
    divergence_color: ColorType = "C3", alpha: float = 0.2, 
    divergence_alpha: float = 0.4, **kwargs) -> plt.Axes:
    """Plots a scatter plot of two parameters, highlighting divergences.
    
    Divergences are plotted in a different color. Divergences occur 
    when the potential-energy landscape is too steep to effectively
    sample, causing the sampler to remain in place, potentially 
    indicating poor sampling.

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    var1, var2 : str
        The variables to be plotted.
    ax : plt.Axes, optional
        The MatPlotLib axes to plot on. If none provided, axes will be
        automatically created. 
    gauss_id : int, default=1
        For the parameters associated with a gaussian fit (r0, w, a),
        which gaussian to plot. E.g., if var1 is set to w and gauss_id
        is set to 2, the widths of the second gaussian will be plotted.
        Counting starts at 1.
    color : ColorType, default="C2"
        The color to plot the non-divergent points.
    divergence_color : ColorType, default="C3"
        The color to plot the divergent points.
    alpha : float, default=0.2
        The transparency of the non-divergent points.
    divergence_alpha : float, default=0.4
        The transparency of the divergent points.
    **kwargs : dict, optional
        Keyword arguments to pass to plt.plot.

    Returns
    -------
    ax : plt.Axes

    See Also
    --------
    plt.plot
    """
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        # creates an ax object if not provided
    gauss_id -= 1 # to fix off-by-one error
    # extract points for each variable
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
    # plots all the points first
    ax.plot(v1, v2, ".", color=color, alpha=alpha, **kwargs)
    # then, plots the divergent points in divergence_color & larger
    divergent = az.extract(trace, group="sample_stats", var_names=["diverging"])
    ax.plot(v1[divergent], v2[divergent], "o", color=divergence_color, 
            alpha=divergence_alpha, **kwargs)
    ax.set_xlabel(_replace_labels(xlabel))
    ax.set_ylabel(_replace_labels(ylabel))
    ax.set_title("scatter plot with divergences between %s and %s" 
                 % (_replace_labels(xlabel), _replace_labels(ylabel)))
    return ax

def pairplot_condition(
    trace: az.InferenceData, var1: str, var2: str, ax: plt.Axes = None, 
    gauss_id: int = 1, criterion: str = None, threshold: float = None, 
    color_greater: ColorType = "dodgerblue", 
    color_lesser: ColorType = "hotpink", alpha_greater: float = 0.2, 
    alpha_lesser: float = 0.2, **kwargs) -> plt.Axes:
    """Plots two parameters in two groups based on a criterion.
    
    The criteria can be found in the keys of trace.sample_stats, 
    including tree depth and step size. 

    Parameters
    ----------
    trace : az.InferenceData
        The trace to be read.
    var1, var2 : str
        The variables to be plotted.
    ax : plt.Axes, optional
        The MatPlotLib axes to plot on. If none provided, axes will be
        automatically created. 
    gauss_id : int, default=1
        For the parameters associated with a gaussian fit (r0, w, a),
        which gaussian to plot. E.g., if var1 is set to w and gauss_id
        is set to 2, the widths of the second gaussian will be plotted.
        Counting starts at 1.
    color_greater : ColorType, default="dodgerblue"
        The color to plot the samples greater than the condition.
    color_lesser : ColorType, default="hotpink"
        The color to plot the samples less than the condition.
    alpha : float, default=0.2
        The transparency of the samples greater than the condition.
    divergence_alpha : float, default=0.4
        The transparency of the samples less than the condition.
    **kwargs : dict, optional
        Keyword arguments to pass to plt.plot.

    Returns
    -------
    ax : plt.Axes

    See Also
    --------
    plt.plot
    """
    if not ax:
        # creates an ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    # the criterion should be in sample_stats, e.g. tree_depth
    gauss_id -= 1 # to fix off-by-one error
    # gets samples
    v1 = az.extract(trace, var_names=[var1])
    v2 = az.extract(trace, var_names=[var2])
    # gets sample stats for criterion
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
    for i in range(len(v1)):
        if stats[i] > threshold:
            ax.plot(v1[i], v2[i], ".", color=color_greater, 
            alpha=alpha_greater, **kwargs)
        else:
            ax.plot(v1[i], v2[i], ".", color=color_lesser, 
            alpha=alpha_lesser, **kwargs)
    ax.set_xlabel(_replace_labels(xlabel))
    ax.set_ylabel(_replace_labels(ylabel))
    ax.set_title("scatter plot between %s and %s split at %s = %s" 
                 % (_replace_labels(xlabel), _replace_labels(ylabel), 
                    criterion, threshold))
    return ax

def plot_hist(
    trace: az.InferenceData, var: str, ax: plt.Axes = None, 
    combine: bool = False, gauss_id: int = 1, **kwargs) -> plt.Axes:
    """Plots a histogram of a parameter's values.
    
    Parameters
    ----------
    trace : az.InferenceData
        The trace to read.
    var : str
        The varaible to plot.
    ax : plt.Axes, optional
        The MatPlotLib axes to plot on. If none provided, axes will be
        automatically created. 
    combine : bool, default=False
        For the parameters associated with a gaussian fit (r0, w, a),
        whether or not to combine all gaussians. E.g., if set to True,
        the r0, w, and a values for all gaussians will be combined.
    gauss_id : int, default=1
        For the parameters associated with a gaussian fit (r0, w, a),
        which gaussian to plot. E.g., if var1 is set to w and gauss_id
        is set to 2, the widths of the second gaussian will be plotted.
        Counting starts at 1. Only if combine is False.
    **kwargs : dict, optional
        Keyword arguments to pass to plt.hist.
    
    Returns
    -------
    ax : plt.Axes

    See Also
    --------
    plt.hist
        
    """
    if not ax:
        # creates an ax object if not provided
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    gauss_id -= 1 # to fix off-by-one error
    xlabel = var
    # gets samples
    v = az.extract(trace, var_names=[var])
    if len(v.dims) > 1:
        if combine: # for r0, w, a, combine into one histogram
            v = v.unstack().stack(stacked=["draw",...]) # combines into one list
        else:
            v = v[gauss_id] # selects the gaussian chosen in gauss_id
            xlabel = var + "[%s]" % gauss_id
    ax.set_xlabel(_replace_labels(xlabel))
    ax.set_ylabel("number of draws")
    ax.set_title("histogram of %s" % _replace_labels(xlabel))
    ax.hist(v, **kwargs)
    return ax