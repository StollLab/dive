## Plotting 

# # Import modules
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import random
import arviz as az
from IPython.display import display

from .utils import *
from .models import *

def summary(df,model,Vexp,t,r,nDraws = 100):

    # df = replaceLabels(df)
    VarNames = df.varnames

    if 'r0' in VarNames:
        Model = 'Gaussian'
        if df['r0'].ndim == 1:
            nGaussians = 1
        else:
            nGaussians = df['r0'].shape[1]

        nSamples = df['r0'].shape[0]

        display('figure out what labels are needed')
        Vars = ["k", "lamb","V0","sigma","tau","delta"]  
    else:
        Model = 'Regularization'
        nSamples = df['P'].shape[0]
        Vars = ["k", "lamb","V0","sigma","tau","delta"]     
    

    with model:
        summary = az.summary(df,var_names=Vars)

    summary.index = betterLabels(summary.index.values)
    display(summary)


    fig, axs = plt.subplots(2, 3)
    fig.set_figheight(7)
    fig.set_figwidth(11)

    axs = np.reshape(axs,(6,))

    for i in range(len(Vars)):
        az.plot_kde(df[Vars[i]],ax = axs[i])
        bottom, top = axs[i].get_ylim()
        # axs[i].vlines(np.mean(df[Vars[i]]),bottom,top, color = 'black')
        axs[i].set_xlabel(betterLabels(Vars[i]))
        axs[i].yaxis.set_ticks([])

    # plt.grid()
    fig.tight_layout()
    plt.show()

    with model:
        axs = az.plot_pair(df,var_names=Vars,kind='kde',figsize=(18,12))

    axs = np.reshape(axs,np.shape(axs)[0]*np.shape(axs)[1])

    for ax in axs:
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if xlabel:
            ax.set_xlabel(betterLabels(xlabel))
        if ylabel:
            ax.set_ylabel(betterLabels(ylabel))

    Ps, Vs, _, _ = drawPosteriorSamples(df,r,t,nDraws)
    plotMCMC(Ps,Vs,Vexp,t,r)


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
    else:
        Model = 'Regularization'
        nSamples = df['P'].shape[0]

    idxSamples = random.sample(range(nSamples),nDraws)
    Ps = []
    Vs = []

    K = dipolarkernel(t,r)

    k_vecs = df['k'][idxSamples]
    lamb_vecs = df['lamb'][idxSamples]
    V0_vecs = df['V0'][idxSamples]

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

    return Ps, Vs, t, r

def plotMCMC(Ps,Vs,Vdata,t,r):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(11)

    
    for V,P in zip(Vs,Ps):
        ax1.plot(t, V, color = '#3F60AE', alpha=0.2)
        ax1.plot(t, Vdata-V, color = '#3F60AE', alpha=0.2)
        ax2.plot(r, P, color = '#3F60AE', alpha=0.2)
    ax1.plot(t, Vdata , color = 'black')
    ax1.hlines(0,min(t),max(t), color = 'black')

    ax1.set_xlabel('$t$ (µs)')
    ax1.set_ylabel('$V$ (arb.u)')
    ax1.set_xlim((min(t), max(t)))
    ax1.set_title('time domain and residuals')

    ax2.set_xlabel('$r$ (nm)')
    ax2.set_ylabel('$P$ (nm$^{-1}$)')
    ax2.set_xlim((min(r), max(r)))
    ax2.set_title('distance domain')

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
