"""Deterministic departure from the initial mode, in the existing figure style."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
import pandas as pd
import matplotlib.pyplot as plt
import rotating_potential_analysis as p
from deterministic_pathway_diagnostic import OUT
p.style()
fine=pd.read_csv(OUT/'trace-phase0.000000-above-dt0.000625.csv')
coarse=pd.read_csv(OUT/'trace-phase0.000000-above-dt0.00125.csv')
summary=pd.read_csv(OUT/'summary.csv')
first=summary.query('phase_rad==0 and threshold_side=="above" and dt_s==0.000625').first_vertical_angle_deg.iloc[0]
fig,ax=plt.subplots(1,2,figsize=(6.3,2.3))
colors=['darkgoldenrod','darkslateblue','firebrick']
for key,label,col in zip(['coherent_rms_um','original_mode_rms_um','other_pattern_rms_um'],['Mean H/V','Original mode','Other pattern'],colors):
    ax[0].plot(coarse.theta_deg,coarse[key],color=col,alpha=.3,lw=2)
    ax[0].plot(fine.theta_deg,fine[key],color=col,label=label)
ax[0].set(xlabel=r'$\theta$ (deg)',ylabel=r'RMS displacement ($\mu$m)',xlim=(28.9,75),ylim=(0,2.8))
ax[0].legend(frameon=False,fontsize=8,loc='upper left')
for key,label,col,ls in zip(['minimum_vertical_actual_um','minimum_vertical_mean_plus_mode_um','minimum_vertical_mode_only_um'],
                           ['Actual path','Means + mode','Mode only'],['firebrick','darkgoldenrod','darkslateblue'],['-','--',':']):
    ax[1].plot(fine.theta_deg,fine[key],color=col,ls=ls,label=label)
ax[1].axhline(0,color='0.5',lw=.8)
ax[1].set(xlabel=r'$\theta$ (deg)',ylabel=r'Minimum vertical $s$ ($\mu$m)',xlim=(28.9,75),ylim=(-5.6,5.7))
ax[1].legend(frameon=False,fontsize=8,loc='lower left')
for i,x in enumerate(ax):
    x.axvline(first,color='0.5',lw=.8,ls='--')
    x.text(0,1.06,f'({chr(97+i)})',transform=x.transAxes,va='bottom')
fig.tight_layout(w_pad=1.4)
for ext in ['pdf','png']:fig.savefig(OUT/f'departure_from_mode.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
