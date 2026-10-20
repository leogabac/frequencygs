"""Noise projection and deterministic closure checks in the paper figure style."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
import pandas as pd
import matplotlib.pyplot as plt
import rotating_potential_analysis as p
from projected_mode_noise import OUT
p.style();purple='darkslateblue';gold='darkgoldenrod'
linear=pd.read_csv(OUT/'linear_noise_projection-tau8.csv')
coarse=pd.read_csv(OUT/'trace-nz33-ntheta181-dt0.00025-n2048-seed81010.csv')
fine=pd.read_csv(OUT/'trace-nz65-ntheta361-dt0.000125-n2048-seed91010.csv')
check=pd.read_csv(OUT/'deterministic_closure_checks-nz65-ntheta361.csv').iloc[:2]
fig,ax=plt.subplots(1,2,figsize=(6.3,2.3))
ax[0].plot(linear.theta_deg,linear.quadrature_variance_exact_block_um2,color='0.5',ls=':',label='Full block, linear')
ax[0].plot(linear.theta_deg,linear.quadrature_variance_two_coordinate_um2,color=purple,ls='--',label='Fixed pair, linear')
ax[0].plot(coarse.t_s*90/8,(coarse.var_z0+coarse.var_z1)/2,color=gold,alpha=.4,lw=2)
ax[0].plot(fine.t_s*90/8,(fine.var_z0+fine.var_z1)/2,color=gold,label='Fixed pair, nonlinear')
ax[0].set(xlabel=r'$\theta$ (deg)',ylabel=r'Quadrature variance ($\mu$m$^2$)',xlim=(20,50),ylim=(0,18))
ax[0].legend(frameon=False,fontsize=8,loc='upper left')
ax[1].plot(check.initial_max_vertical_amplitude_um,check.unrestricted_final_vertical_defects,'o',color=purple,label='Free continuation')
ax[1].plot(check.initial_max_vertical_amplitude_um,check.two_quadrature_final_vertical_defects,'s',color=gold,label='Fixed pair')
ax[1].set(xlabel=r'Initial vertical amplitude ($\mu$m)',ylabel='Retained vertical defects',xlim=(.80,1.05),ylim=(-1.5,22))
ax[1].set_yticks([0,5,10,15,20]);ax[1].legend(frameon=False,fontsize=8,loc='upper left')
for i,x in enumerate(ax):x.text(0,1.06,f'({chr(97+i)})',transform=x.transAxes,va='bottom')
fig.tight_layout(w_pad=1.4)
for ext in ['pdf','png']:fig.savefig(OUT/f'noise_and_closure.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
