"""Plot the corrected coupled model alongside saved motion and rare crossings."""
import coupled_relaxation_analysis as c
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def save(fig,stem):
    for ext in ['pdf','png']:fig.savefig(c.OUT/f'{stem}.{ext}',dpi=300,bbox_inches='tight')

def main():
    c.p.style()
    summaries=sorted(c.OUT.glob('summary-*-nt901.csv'))
    assert summaries,'Run coupled_relaxation_analysis first'
    summary=c.pd.read_csv(summaries[-1])
    paths=c.pd.read_csv(c.OUT/summaries[-1].name.replace('summary-','paths-'))
    observed=c.pd.read_csv(c.p.REPO/'data/analysis/left-boundary-timescales/averaged_motion.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.6/2.54,5.3/2.54),layout='constrained')
    for ax,coordinate,obscol,label,letter in zip(axes,['s_h_um','s_v_um'],['Rh','Rv'],[r'$R_h/d$',r'$R_v/d$'],['a)','b)']):
        for model,color,ls,name in [('fixed verticals','.6','--','fixed'),('coupled relaxation','darkslateblue','-','coupled')]:
            g=paths.query('tau_s==8 and model==@model')
            ax.plot(g.theta_deg,abs(g[coordinate]-5)/10,color=color,ls=ls,label=name)
        for temp,color,marker,name in [(.1,'black','v','cold data'),(300,'darkgoldenrod','o','warm data')]:
            g=observed.query('total_time==8 and temperature_K==@temp')
            ax.plot(g.t*90/8,g[obscol],marker,ms=1.8,markevery=5,color=color,label=name)
        ax.set_xlim(0,90);ax.set_xticks([0,45,90]);ax.set_xlabel(r'$\theta\,[\mathrm{deg}]$')
        ax.set_ylabel(label,labelpad=0);ax.text(.04,.95,letter,transform=ax.transAxes,va='top')
    axes[0].legend(frameon=False,fontsize=5.5,handlelength=1,loc='upper left',bbox_to_anchor=(0,.88))
    save(fig,'coupled_motion_comparison')
    fig,ax=plt.subplots(figsize=(8.6/2.54,5.3/2.54),layout='constrained')
    old=c.pd.read_csv(c.v.OUT/'probabilities-dx0.005-dt0.005.csv').query('temperature_K==300').sort_values('omega_rad_s')
    ax.plot(old.omega_rad_s,old.p_vertical_final,'--',color='.6',label='previous fixed background')
    for model,color,name in [('fixed verticals','darkgoldenrod','corrected fixed background'),('coupled relaxation','darkslateblue','coupled relaxation')]:
        g=summary.query('model==@model').sort_values('omega_rad_s')
        if model=='coupled relaxation' and (c.OUT/'boundary_evaluations.csv').exists():
            root=c.pd.read_csv(c.OUT/'boundary_evaluations.csv')
            root=root.query('dx_um==.005 and dt_s==.005 and nt==901')
            g=c.pd.concat([g,root]).drop_duplicates('tau_s').sort_values('omega_rad_s')
        ax.plot(g.omega_rad_s,g.p_vertical_final,'o-',color=color,ms=2,label=name)
    obs=c.pd.read_csv(c.p.REPO/'data/analysis/left-boundary-timescales/timescales.csv').query('temperature_K==300 and fv_final>0 and omega<.25')
    ax.plot(obs.omega,obs.fv_final,'v',color='black',ms=3,label='warm data')
    ax.axhline(.01,ls='--',lw=.7,color='.4');ax.set_yscale('log');ax.set_ylim(1e-8,1);ax.set_xlim(0,.22)
    ax.set_xlabel(r'$\omega\,[\mathrm{rad/s}]$');ax.set_ylabel(r'$p_v(t_{\rm obs})$',labelpad=0)
    ax.legend(frameon=False,fontsize=6,loc='lower left',handlelength=1)
    save(fig,'coupled_warm_probability')
    print('Saved two new figures:',c.OUT)

if __name__=='__main__':main()
