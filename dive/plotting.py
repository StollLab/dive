## Plotting 

# # Import modules
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import random
import arviz as az
from IPython.display import display
import deerlab as dl

from .utils import *
from .models import *

def summary(df,model,Vexp,t,r,nDraws = 100, Pref = None):

    # df = replaceLabels(df)
    VarNames = df.varnames

    if 'r0' in VarNames:
        Model = 'Gaussian'
        if df['r0'].ndim == 1:
            nGaussians = 1
        else:
            nGaussians = df['r0'].shape[1]

        nSamples = df['r0'].shape[0]

        if nGaussians == 1:
            Vars = ["r0", "w","k","lamb","V0","sigma"]  
        else:
            Vars = ["r0", "w","a","k","lamb","V0","sigma"]

    elif 'k' not in VarNames:
        if 'sigma' not in VarNames:
            Model = 'Edwards'
            nSamples = df['P'].shape[0]
            if "V0" in VarNames:
                Vars = ["V0","delta",'lg_alpha']
            else:
                Vars = ["delta",'lg_alpha']
        else:
            Model = 'ExpandedEdwards'
            nSamples = df['P'].shape[0]
            Vars = ["V0","sigma","tau","delta",'lg_alpha']

    else:
        Model = 'Regularization'
        nSamples = df['P'].shape[0]
        Vars = ["k", "lamb","V0","sigma","tau","delta",'lg_alpha']     
    
    with model:
        summary = az.summary(df,var_names=Vars)

    summary.index = betterLabels(summary.index.values)
    display(summary)
  

    nrows = int(np.ceil(len(Vars)/4))

    fig, axs = plt.subplots(nrows, len(Vars))
    axs = np.reshape(axs,(len(Vars),))
    
    height = nrows*3.5
    width = 11

    fig.set_figheight(height)
    fig.set_figwidth(width)

    for i in range(len(Vars)):
        az.plot_kde(df[Vars[i]],ax = axs[i])
        bottom, top = axs[i].get_ylim()
        # axs[i].vlines(np.mean(df[Vars[i]]),bottom,top, color = 'black')
        axs[i].set_xlabel(betterLabels(Vars[i]))
        axs[i].yaxis.set_ticks([])

    fig.tight_layout()
    plt.show()

    if len(Vars) < 3:
        corwidth = 7
        corheight = 7
    else:
        corwidth = 11
        corheight = 11

    with model:
        axs = az.plot_pair(df,var_names=Vars,kind='kde',figsize=(corwidth,corheight))

    if len(Vars) > 2:
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

    plt.show()

    Ps, Vs, _, _ = drawPosteriorSamples(df,r,t,nDraws)
    plotMCMC(Ps, Vs, Vexp, t, r, Pref)


def betterLabels(x):

    if type(x) == str:
        x = LabelLookup(x)
    else:
        for i in range(len(x)):
            x[i] = LabelLookup(x[i])

    return x

def LabelLookup(input):
    if input == "lamb":
        return "λ"
    elif input == "sigma":
        return "σ"
    elif input == "delta":
        return "δ"
    elif input == "tau":
        return "τ" 
    elif input == "V0":
        return "V₀"
    elif input == "r0":
        return "r₀"
    elif input == "alpha":
        return "α"
    elif input == "lg_alpha":
        return "lg(α)"
    else:
        return input

def drawPosteriorSamples(df, r = np.linspace(2, 10,num = 300), t = np.linspace(0,3,num = 200), nDraws = 100):
    VarNames = df.varnames

    if 'r0' in VarNames:
        Model = 'Gaussian'
        if df['r0'].ndim == 1:
            nGaussians = 1
        else:
            nGaussians = df['r0'].shape[1]

        nSamples = df['r0'].shape[0]

    elif 'k' not in VarNames:
        if 'sigma' not in VarNames:
            Model = 'Edwards'
            nSamples = df['P'].shape[0]
        else:
            Model = 'ExpandedEdwards'
            nSamples = df['P'].shape[0]

    else:
        Model = 'Regularization'
        nSamples = df['P'].shape[0]

    idxSamples = random.sample(range(nSamples),nDraws)
    Ps = []
    Vs = []

    K = dl.dipolarkernel(t,r,integralop=True)
    K[:,0] = 2*K[:,0]
    K[:,-1] = 2*K[:,-1]

    if Model != 'Edwards':

        if 'V0' in VarNames:
            V0_vecs = df['V0'][idxSamples]

        if Model != 'ExpandedEdwards' and Model != 'Edwards':
            k_vecs = df['k'][idxSamples]
            lamb_vecs = df['lamb'][idxSamples]

    if Model == 'Gaussian':
        r0_vecs = df['r0'][idxSamples]
        w_vecs = df['w'][idxSamples]
        if nGaussians == 1:
            a_vecs = np.ones_like(idxSamples)
        else:
            a_vecs = df['a'][idxSamples]

        for iDraw in range(nDraws):
            P = dd_gauss(r,r0_vecs[iDraw],w_vecs[iDraw],a_vecs[iDraw])
            Ps.append(P)
            
            B = bg_exp(t,k_vecs[iDraw])
            F = np.dot(K,P)
            Vs.append(deerTrace(F,B,V0_vecs[iDraw],lamb_vecs[iDraw]))

    elif Model == 'Regularization':
 
        for iDraw in range(nDraws):
            P = df['P'][idxSamples[iDraw]]
            Ps.append(P)

            B = bg_exp(t,k_vecs[iDraw])
            F = np.dot(K,P)
            Vs.append(deerTrace(F,B,V0_vecs[iDraw],lamb_vecs[iDraw]))

    elif Model == 'Edwards':

        for iDraw in range(nDraws):
            P = df['P'][idxSamples[iDraw]]
            Ps.append(P)
            if 'V0' not in VarNames:
                Vs.append(np.dot(K,P))
            else:  
                Vs.append(V0_vecs[iDraw]*np.dot(K,P))

    elif Model == 'ExpandedEdwards':

        for iDraw in range(nDraws):
            P = df['P'][idxSamples[iDraw]]
            Ps.append(P)
            Vs.append(V0_vecs[iDraw]*np.dot(K,P))

    return Ps, Vs, t, r

def plotMCMC(Ps,Vs,Vdata,t,r, Pref = None):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(11)

    if min(Vs[0])<0.2:
        residuals_offset = -max(Vs[0])/3
    else:
        residuals_offset = 0

    for V,P in zip(Vs,Ps):
        ax1.plot(t, V, color = '#3F60AE', alpha=0.2)
        ax1.plot(t, Vdata-V+residuals_offset, color = '#3F60AE', alpha=0.2)
        ax2.plot(r, P, color = '#3F60AE', alpha=0.2)
    ax1.plot(t, Vdata , color = 'black')
    ax1.hlines(residuals_offset,min(t),max(t), color = 'black')

    ax1.set_xlabel('$t$ (µs)')
    ax1.set_ylabel('$V$ (arb.u)')
    ax1.set_xlim((min(t), max(t)))
    ax1.set_title('time domain and residuals')

    ax2.set_xlabel('$r$ (nm)')
    ax2.set_ylabel('$P$ (nm$^{-1}$)')
    ax2.set_xlim((min(r), max(r)))
    ax2.set_title('distance domain')

    if Pref is not None:
        ax2.plot(r, Pref , color = 'black')

    plt.grid()

    plt.show()




# def drawFromPrior(model):
#     """
#     Draw from the prior definition
#     """

# # Load results

# # Posteriors
# posts = pm.plot_posterior(tr, ['r0'],credible_interval=0.95,ref_val=3)
# posts_pl = plt.gcf()
# posts_pl.savefig('2gauss_r0ref1.svg', bbox_inches='tight')

# traces = pm.traceplot(tr, ['r0'])
# traces_pl = plt.gcf()
# traces_pl.savefig('2gauss_r0traces.pdf', bbox_inches='tight')

# # Correlation plots 

# df = pm.trace_to_dataframe(trace)
# sb.set(style="white")
# plt.figure(figsize=(6, 6))
# g = sb.PairGrid(df, diag_sharey=False)
# g.map_lower(sb.kdeplot)
# g.map_diag(sb.kdeplot, lw=3)
# for i, j in zip(*np.triu_indices_from(g.axes, 1)):
#     g.axes[i, j].set_visible(False)
# g.axes[0,0].set_xlim((np.amin(r),np.amax(r)))
# g.axes[0,1].set_xlim((0,1))
# g.axes[0,2].set_xlim((0,1))
# g.axes[0,3].set_xlim((0,6))
# g.axes[0,4].set_xlim((15,25))
# g.axes[0,5].set_xlim((0,Vmax))

# corr_plots2 = plt.gcf()
# corr_plots2.savefig('1gauss_corrplots_seaborn.png', bbox_inches = 'tight')
