"""Compare instantaneous Kramers rates with the existing driven probability result."""
import kramers_transition_analysis as k
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    k.c.p.style()
    summary=k.pd.read_csv(k.OUT/'summary-T300-dx0.005-nt401.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.6/2.54,6.1/2.54),layout='constrained')
    for name,color in [('fixed','darkgoldenrod'),('coupled','darkslateblue')]:
        data=k.pd.read_csv(k.OUT/f'{name}-T300-tau8-dx0.005-nt401.csv')
        axes[0].plot(data.theta_deg,data.barrier_forward_kBT,color=color,label=name)
    axes[0].set_xlim(30,65);axes[0].set_ylim(0,80)
    axes[0].set_xlabel(r'$\theta\,[\mathrm{deg}]$');axes[0].set_ylabel(r'$\Delta U/k_BT$',labelpad=0)
    axes[0].legend(frameon=False,fontsize=6,loc='upper right',handlelength=1)
    data=summary.query('model=="coupled relaxation"').sort_values('omega_rad_s')
    axes[1].plot(data.omega_rad_s,data.p_final_harmonic,'o--',color='darkgoldenrod',ms=2,label='harmonic rate')
    axes[1].plot(data.omega_rad_s,data.p_final_integral,'s--',color='darkslateblue',ms=2,label='integral rate')
    exact=k.pd.read_csv(k.c.OUT/'summary-dx0.005-dt0.005-nt901.csv').query('model=="coupled relaxation"').sort_values('omega_rad_s')
    axes[1].plot(exact.omega_rad_s,exact.p_vertical_final,'o-',color='.5',ms=2,label='full probability')
    obs=k.pd.read_csv(k.c.p.REPO/'data/analysis/left-boundary-timescales/timescales.csv').query('temperature_K==300 and fv_final>0 and omega<.25')
    axes[1].plot(obs.omega,obs.fv_final,'v',color='black',ms=2.5,label='warm data')
    axes[1].axhline(.01,color='.4',ls=':',lw=.6)
    axes[1].set_yscale('log');axes[1].set_ylim(1e-15,1);axes[1].set_yticks([1e-15,1e-10,1e-5,1])
    axes[1].set_xlim(0,.22);axes[1].set_xticks([0,.1,.2])
    axes[1].set_xlabel(r'$\omega\,[\mathrm{rad/s}]$',fontsize=9)
    axes[1].set_ylabel(r'$p_v(t_{\rm obs})$',labelpad=0)
    axes[1].legend(frameon=False,fontsize=5.2,loc='lower left',handlelength=.9)
    for ax,letter in zip(axes,['a)','b)']):ax.text(.04,.96,letter,transform=ax.transAxes,va='top')
    for ext in ['pdf','png']:fig.savefig(k.OUT/f'kramers_comparison.{ext}',dpi=300,bbox_inches='tight')
    print('Saved Kramers comparison:',k.OUT)

if __name__=='__main__':main()
