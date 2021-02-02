# ## Plotting 

# # Import modules
# import pymc3 as pm
# import matplotlib.pyplot as plt
# import seaborn as sb

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
