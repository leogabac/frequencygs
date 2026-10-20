"""Compact timescale figure for the separate formal theory note."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
import pandas as pd
import matplotlib.pyplot as plt
import rotating_potential_analysis as p
import collective_stability_analysis as a
from pathway_selection_theory import OUT
p.style()
purple='darkslateblue';gold='darkgoldenrod'
f=pd.read_csv(OUT/'bare_trap_survival.csv').query('dx_um==0.005 and t_s<=30')
tr=pd.read_csv(a.OUT/'trace-tau8-n3600.csv')
fig,ax=plt.subplots(1,2,figsize=(6.3,2.3))
ax[0].plot(f.t_s,f.p_first_hill_single,color=purple,label='One particle')
ax[0].plot(f.t_s,f.p_first_hill_among_100,color=gold,label='First among 100')
ax[0].axvline(8,color='0.5',ls=':',lw=.8)
ax[0].set(xlabel=r'$t$ (s)',ylabel='Hill-reaching probability',xlim=(0,30),ylim=(0,1.03))
ax[0].legend(frameon=False,fontsize=8,loc='upper left')
ax[1].plot(tr.theta_deg,tr.largest_eigenvalue_per_s,color=purple,label='All along-trap modes')
ax[1].plot(tr.theta_deg,tr.largest_coherent_HV_eigenvalue_per_s,color=gold,label='Coherent H/V subspace')
ax[1].axhline(0,color='0.5',lw=.7)
ax[1].axvspan(28.9,46.97377272095176,color=purple,alpha=.09)
ax[1].set(xlabel=r'$\theta$ (deg)',ylabel=r'$\lambda_{\max}$ (s$^{-1}$)',xlim=(20,60),ylim=(-5.5,3.5))
ax[1].legend(frameon=False,fontsize=8,loc='lower left')
for i,x in enumerate(ax):
    x.text(0,1.06,f'({chr(97+i)})',transform=x.transAxes,va='bottom')
fig.tight_layout(w_pad=1.4)
for ext in ['pdf','png']:fig.savefig(OUT/f'timescale_comparison.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
