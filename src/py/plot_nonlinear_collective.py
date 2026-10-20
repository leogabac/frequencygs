"""Compare retained nonlinear crossings with existing warm/cold observations."""
import nonlinear_collective_analysis as a
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(a.OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def main():
    a.p.style();selected=pd.read_csv(a.OUT/'selected_results.csv')
    obs=pd.read_csv(a.p.REPO/'data/analysis/left-boundary-timescales/averaged_motion.csv')
    table=pd.read_csv(a.p.REPO/'data/analysis/left-boundary-timescales/timescales.csv')
    fig,ax=plt.subplots(figsize=(8.6/2.54,5.6/2.54),layout='constrained')
    for tau,color in [(7,'.5'),(8,'darkslateblue'),(9,'darkgoldenrod')]:
        r=selected.query('temperature_K==300 and tau_s==@tau')
        if r.empty:continue
        row=r.iloc[0];path=a.OUT/row.source.replace('ensemble-','trace-').replace('.npz','.csv');f=pd.read_csv(path)
        ax.plot(f.theta_deg,f.p_vertical,color=color,label=rf'$\tau={tau}\,\mathrm s$')
        ax.fill_between(f.theta_deg,np.maximum(f.p_vertical-1.96*f.p_vertical_se,0),f.p_vertical+1.96*f.p_vertical_se,color=color,alpha=.12,lw=0)
        g=obs.query('temperature_K==300 and total_time==@tau')
        ax.plot(g.t*90/tau,g.fv,'o',ms=1.8,markevery=10,color=color)
    ax.set_xlim(35,90);ax.set_ylim(0,.04);ax.set_xlabel(r'$\theta\,[\mathrm{deg}]$');ax.set_ylabel(r'$p_v(t)$',labelpad=0)
    ax.legend(frameon=False,fontsize=6,loc='upper left');ax.text(.98,.05,'lines: nonlinear model\npoints: existing data',ha='right',va='bottom',fontsize=6,transform=ax.transAxes)
    save(fig,'nonlinear_retained_switches')
    fig,axes=plt.subplots(1,2,figsize=(8.6/2.54,5.6/2.54),layout='constrained')
    for ax,temp,letter in zip(axes,[300,.1],['a)','b)']):
        g=selected.query('temperature_K==@temp').sort_values('omega_rad_s')
        if len(g):ax.errorbar(g.omega_rad_s,g.p_vertical_final,yerr=[g.p_error_minus,g.p_error_plus],fmt='o-',color='darkslateblue',ms=2,capsize=2,label='model')
        d=table.query('temperature_K==@temp')
        ax.plot(d.omega,d.fv_final,'v',color='black',ms=3,label='data')
        ax.axhline(.01,ls='--',lw=.7,color='.5');ax.set_ylim(0,.04 if temp==300 else .2)
        ax.set_xlim((.16,.24) if temp==300 else (.035,.06));
        if temp==300:ax.set_xticks([.17,.20,.23]);
        ax.set_xlabel(r'$\omega\,[\mathrm{rad/s}]$');ax.set_ylabel(r'$p_v(t_{\rm obs})$',labelpad=0)
        ax.text(.96,.95,letter,ha='right',va='top',transform=ax.transAxes);ax.set_title(rf'$T={temp:g}\,\mathrm K$',fontsize=7)
    axes[0].legend(frameon=False,fontsize=6,loc='upper left');save(fig,'nonlinear_boundary_comparison')
    fig,axes=plt.subplots(1,2,figsize=(8.6/2.54,5.4/2.54),layout='constrained')
    row=selected.query('temperature_K==300 and tau_s==8').iloc[0]
    f=pd.read_csv(a.OUT/row.source.replace('ensemble-','trace-').replace('.npz','.csv'))
    g=obs.query('temperature_K==300 and total_time==8')
    for ax,col,letter in zip(axes,['Rh','Rv'],['a)','b)']):
        ax.plot(f.theta_deg,f[col],color='darkslateblue',label='model');ax.plot(g.t*90/8,g[col],'o',ms=1.8,markevery=10,color='darkgoldenrod',label='warm data')
        ax.set_xlabel(r'$\theta\,[\mathrm{deg}]$');ax.set_ylabel(r'$R_'+col[-1]+r'/d$',labelpad=0);ax.set_xticks([0,45,90]);ax.set_xlim(0,90)
        ax.text(.96,.95,letter,ha='right',va='top',transform=ax.transAxes)
    axes[0].legend(frameon=False,fontsize=6,loc='upper left');save(fig,'nonlinear_mean_motion')
if __name__=='__main__':main()
