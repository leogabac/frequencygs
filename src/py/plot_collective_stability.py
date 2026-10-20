"""Plot the linear diagnostic; do not identify a Gaussian tail with escape."""
import collective_stability_analysis as a
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    a.p.style();out=a.OUT
    f=pd.read_csv(out/'trace-tau8-n3600.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.6/2.54,5.6/2.54),layout='constrained')
    ax=axes[0]
    ax.plot(f.theta_deg,f.largest_eigenvalue_per_s,color='darkslateblue',label='all modes')
    ax.plot(f.theta_deg,f.largest_coherent_HV_eigenvalue_per_s,'--',color='darkgoldenrod',label='coherent H/V')
    ax.axhline(0,color='.5',lw=.6);ax.set_ylim(-3,3.2)
    ax.set_ylabel(r'$\lambda_{\max}\,[\mathrm{s}^{-1}]$',labelpad=0)
    ax.legend(frameon=False,fontsize=5.5,handlelength=1)
    ax=axes[1]
    ax.plot(f.theta_deg,f.sigma_v_300K_um,color='darkslateblue',label='collective')
    ax.plot(f.theta_deg,f.sigma_v_diagonal_300K_um,'--',color='darkgoldenrod',label='diagonal control')
    ax.plot(f.theta_deg,np.maximum(5-f.s_v_um,0),':',color='.35',label='wall distance')
    ax.set_ylabel(r'$\sigma_v\,[\mu\mathrm m]$',labelpad=0);ax.set_ylim(0,2.6)
    ax.legend(frameon=False,fontsize=5.5,handlelength=1,loc='upper left')
    for ax,letter in zip(axes,['a)','b)']):
        ax.set_xlim(0,60);ax.set_xticks([0,30,60]);ax.set_xlabel(r'$\theta\,[\mathrm{deg}]$')
        ax.text(.96,.70,letter,ha='right',va='top',transform=ax.transAxes)
    for ext in ['png','pdf']:fig.savefig(out/f'collective_stability.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.6/2.54,5.3/2.54),layout='constrained')
    for tau,color in [(7,'.5'),(8,'darkslateblue'),(9,'darkgoldenrod')]:
        g=pd.read_csv(out/f'trace-tau{tau}-n3600.csv')
        ax.plot(g.theta_deg,g.gaussian_vertical_tail_300K,color=color,label=rf'$\tau={tau}\,\mathrm s$')
    obs=pd.read_csv(a.p.REPO/'data/analysis/left-boundary-timescales/averaged_motion.csv')
    for tau,color in [(7,'.5'),(8,'darkslateblue'),(9,'darkgoldenrod')]:
        g=obs.query('temperature_K==300 and total_time==@tau')
        ax.plot(g.t*90/tau,g.fv,'o',ms=1.8,markevery=5,color=color)
    ax.set_xlim(25,60);ax.set_ylim(1e-6,.1);ax.set_yscale('log')
    ax.set_xlabel(r'$\theta\,[\mathrm{deg}]$');ax.set_ylabel(r'$p_v$',labelpad=0)
    ax.legend(frameon=False,fontsize=6,handlelength=1,loc='upper left')
    ax.text(.97,.03,'lines: Gaussian instantaneous tail\npoints: measured crossing fraction',transform=ax.transAxes,fontsize=5.5,ha='right',va='bottom')
    for ext in ['png','pdf']:fig.savefig(out/f'collective_gaussian_tail.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    model=a.c.CoupledModel();lattice=a.Lattice(model);sol=model.solve(8)
    theta=np.deg2rad(44);q=sol.sol(8*44/90)
    lam,V=np.linalg.eigh(lattice.blocks(q,theta));kk,jj=np.unravel_index(lam.argmax(),lam.shape)
    vec=V[kk,:,jj];vec*=np.exp(-1j*np.angle(vec[np.argmax(abs(vec))]))
    mode=np.empty(200)
    for cc,sites in enumerate(lattice.order):mode[sites]=(lattice.phase[kk,cc]*vec).real
    mode/=np.max(abs(mode));xy=model.centers[:,:2]
    table=pd.DataFrame({'id':model.state.id,'x_um':xy[:,0],'y_um':xy[:,1],'horizontal':model.is_h,'delta_s_normalized':mode,
                        'delta_x_normalized':mode*lattice.axis[:,0],'delta_y_normalized':mode*lattice.axis[:,1]})
    table.to_csv(out/'dominant_mode-theta44-tau8.csv',index=False)
    (out/'dominant_mode.json').write_text(a.json.dumps({'theta_deg':44,'tau_s':8,'wavevector_indices':lattice.wave[kk].tolist(),
        'eigenvalue_per_s':float(lam[kk,jj]),'vertical_weight':float(np.sum(abs(vec[~lattice.hmask])**2)),
        'normalization':'max absolute along-trap displacement = 1; arbitrary mode amplitude'},indent=2))
    fig,ax=plt.subplots(figsize=(8.6/2.54,7.8/2.54),layout='constrained')
    for mask,marker,label in [(model.is_h,'s','horizontal'),(~model.is_h,'o','vertical')]:
        ax.scatter(xy[mask,0],xy[mask,1],s=6,marker=marker,color='.65',label=label)
    ax.quiver(xy[:,0],xy[:,1],mode*lattice.axis[:,0]*10,mode*lattice.axis[:,1]*10,mode,
              cmap='coolwarm',clim=(-1,1),angles='xy',scale_units='xy',scale=1,width=.005)
    ax.set_aspect('equal');ax.set_xlabel(r'$x\,[\mu\mathrm m]$');ax.set_ylabel(r'$y\,[\mu\mathrm m]$',labelpad=0)
    ax.set_xticks([0,150,300]);ax.set_yticks([0,150,300]);ax.legend(frameon=False,fontsize=6,loc='upper right')
    for ext in ['png','pdf']:fig.savefig(out/f'collective_mode.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
