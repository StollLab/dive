## Plotting 

# # Import modules
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

def summary(df, model_dic, nDraws=100, Pid=None, Pref=None, GroundTruth=[], rref=None, corrPlot=True, marginalsPlot=True):
    
    # Figure out what Vars are present -----------------------------------------
    possibleVars = ["r0", "w", "a", "k", "lamb", "V0", "sigma", "delta", "lg_alpha"]
    presentVars = df.varnames

    model = model_dic['model']
    Vexp = model_dic['Vexp']
    t = model_dic['t']
    r = model_dic['pars']['r']

    if Pid is not None:
        P0s = loadmat('..\..\data\edwards_testset\distributions_2LZM.mat')['P0']
        rref = np.squeeze(loadmat('..\..\data\edwards_testset\distributions_2LZM.mat')['r0'])
        Pref = P0s[Pid-1,:]
        
    elif Pref is not None:
        if rref is None:
            raise KeyError("If 'Pref' is provided, 'rref' must be provided as well.")

    if GroundTruth:
        plotTruth = True
    else:
        plotTruth = False

    Vars = [Var for Var in possibleVars if Var in presentVars]
    nVars = len(Vars)

    # Print summary for RVs ----------------------------------------------------
    with model:
        summary = az.summary(df, var_names=Vars)
    # replace the labels with their unicode characters before displaying
    summary.index = betterLabels(summary.index.values)
    display(summary)
    
    # Plot marginalized posteriors ---------------------------------------------
    if marginalsPlot:

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
            az.plot_kde(df[Vars[i]], ax=axs[i])
            axs[i].set_xlabel(betterLabels(Vars[i]),fontsize='large')
            axs[i].yaxis.set_ticks([])
            axs[i].grid(axis='x')

            if plotTruth and (Vars[i]in GroundTruth.keys()):
                bottom, top = axs[i].get_ylim()
                axs[i].vlines(GroundTruth[Vars[i]], bottom, top, color='black')

        for i in range(nVars, len(axs)):
            axs[i].axis('off')

        # Clean up figure
        fig.tight_layout()
        plt.show()

    # Pairwise correlation plots ----------------------------------------------
    if corrPlot:
        # determine figure size
        if nVars < 3:
            corwidth = 7
            corheight = 7
        else:
            corwidth = 10
            corheight = 10

        # use arviz library to plot them
        with model:
            axs = az.plot_pair(df, var_names=Vars, kind='kde', figsize=(corwidth,corheight))

        # replace labels with the nicer unicode character versions
        if len(Vars) > 2:
            # reshape axes so that we can loop through them
            axs = np.reshape(axs,np.shape(axs)[0]*np.shape(axs)[1])

            for ax in axs:
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                if xlabel:
                    ax.set_xlabel(betterLabels(xlabel))
                if ylabel:
                    ax.set_ylabel(betterLabels(ylabel))
        else:
            xlabel = axs.get_xlabel()
            ylabel = axs.get_ylabel()
            axs.set_xlabel(betterLabels(xlabel))
            axs.set_ylabel(betterLabels(ylabel))

        # show plot
        plt.show()

    # Posterior sample plot -----------------------------------------------------
    # Draw samples
    Ps, Vs, Bs,  _, _ = drawPosteriorSamples(df,r,t,nDraws)
    # Plot them
    plotMCMC(Ps, Vs, Bs, Vexp, t, r, Pref, rref)

# look up table that contains the strings and their symbols
_table = {
    "lamb": "λ",
    "lamba": "λ",
    "sigma": "σ",
    "delta": "δ",
    "tau": "τ",
    "V0": "V₀",
    "r0": "r₀",
    "alpha": "α",
    "lg_alpha": "lg(α)",
    "w[0]": "w₀",
    "w[1]": "w₁",
    "w[2]": "w₂",
    "w[3]": "w₃",
    "a[0]": "a₀",
    "a[1]": "a₁",
    "a[2]": "a₂",
    "a[3]": "a₃",
    "r0[0]": "r₀,₀",
    "r0[1]": "r₀,₁",
    "r0[2]": "r₀,₂",
    "r0[3]": "r₀,₃",
    "r0\n0": "r₀,₀",
    "r0\n1": "r₀,₁",
    "r0\n2": "r₀,₂",
    "r0\n3": "r₀,₃",
    "a\n0": "a₀",
    "a\n1": "a₁",
    "a\n2": "a₂",
    "a\n3": "a₃",
    "w\n0": "w₀",
    "w\n1": "w₁",
    "w\n2": "w₂",
    "w\n3": "w₃",
}

def betterLabels(x):
    """
    Replace strings with their corresponding (greek) symbols
    """
    if isinstance(x, str):
        return _table.get(x,x)
    else:
        return [_table.get(x_,x_) for x_ in x]

def drawPosteriorSamples(df, r=np.linspace(2, 8,num=200), t=np.linspace(0, 3, num=200), nDraws=100):
    VarNames = df.varnames

    # Determine if a Gaussian model was used and how many iterations were run -------
    if 'r0' in VarNames:
        if df['r0'].ndim == 1:
            nGaussians = 1
        else:
            nGaussians = df['r0'].shape[1]

        nChainSamples = df['r0'].shape[0]

    else:
        nChainSamples = df['P'].shape[0]

    # Generate random indices from chain samples ------------------------------------
    idxSamples = random.sample(range(nChainSamples),nDraws)

    # Draw P's -------------------------------------------------------------------
    Ps = []

    if 'r0' in VarNames:
        r0_vecs = df['r0'][idxSamples]
        w_vecs = df['w'][idxSamples]
        if nGaussians == 1:
            a_vecs = np.ones_like(idxSamples)
        else:
            a_vecs = df['a'][idxSamples]

        for iDraw in range(nDraws):
            P = dd_gauss(r,r0_vecs[iDraw],w_vecs[iDraw],a_vecs[iDraw])
            Ps.append(P)
    else:
        for iDraw in range(nDraws):
            P = df['P'][idxSamples[iDraw]]
            Ps.append(P)

    # Draw corresponding time-domain parameters ---------------------------------
    if 'V0' in VarNames:
        V0_vecs = df['V0'][idxSamples]

    if 'k' in VarNames:
        k_vecs = df['k'][idxSamples]

    if 'lamb' in VarNames:   
        lamb_vecs = df['lamb'][idxSamples]

    # Generate V's from P's and other parameters --------------------------------
    Vs = []
    Bs = []
    K0 = dl.dipolarkernel(t,r,integralop=False)
    dr = r[1] - r[0]

    for iDraw in range(nDraws):
        K_ = copy.copy(K0)

        # The below construction of the kernel only takes into account RVs that were actually sampled
        # During development the model was sometimes run with fixed values for λ, k, or V₀
        if 'lamb' in VarNames: 
            K_ = (1-lamb_vecs[iDraw]) + lamb_vecs[iDraw]*K_

        if 'k' in VarNames:
            B = bg_exp(t,k_vecs[iDraw])
            K_ = K_*B[:, np.newaxis]

        if 'V0' in VarNames:
            K_ = V0_vecs[iDraw]*K_

        K_ = K_*dr
        Bs.append((1-lamb_vecs[iDraw])*B)
        Vs.append(K_@Ps[iDraw])

    return Ps, Vs, Bs, t, r


def plotMCMC(Ps, Vs, Bs, Vdata, t, r, Pref=None, rref=None):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(11)

    if min(Vs[0])<0.2:
        residuals_offset = -max(Vs[0])/3
    else:
        residuals_offset = 0

    for V,P,B in zip(Vs,Ps,Bs):
        residuals = V - Vdata
        rmsd = np.sqrt(np.mean(np.square(residuals)))
        ax1.plot(t, V, color = '#3F60AE', alpha=0.2)
        ax1.plot(t, B, color = '#FCC43F', alpha=0.2)
        ax1.plot(t, residuals+residuals_offset, color = '#3F60AE', alpha=0.2)
        ax2.plot(r, P, color = '#3F60AE', alpha=0.2)
    ax1.scatter(t, Vdata , color = '#BFBFBF', s = 5)
    ax1.hlines(residuals_offset,min(t),max(t), color = 'black')

    ax1.set_xlabel('$t$ (µs)')
    ax1.set_ylabel('$V$ (arb.u.)')
    ax1.set_xlim((min(t), max(t)))
    ax1.set_title('time domain and residuals')

    ax2.set_xlabel('$r$ (nm)')
    ax2.set_ylabel('$P$ (nm$^{-1}$)')
    ax2.set_xlim((min(r), max(r)))
    ax2.set_title('distance domain')

    if Pref is not None:
        ax2.plot(rref, Pref, color='black')

    plt.grid()
    plt.show()
    
    print(f"RMSD: {rmsd}")
    