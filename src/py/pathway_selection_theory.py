"""First-passage baseline and deterministic continuation of collective modes.

New, isolated theory artifacts; no stochastic lattice runs or transition fits.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/frequencygs-numba')
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import dawsn,erf
from scipy.linalg import eigh_tridiagonal,solve_banded
from numba import njit
import rotating_potential_analysis as p
import coupled_relaxation_analysis as c
import collective_horizontal_analysis as h
import collective_stability_analysis as a
import nonlinear_collective_analysis as n
import vertical_competition_analysis as v
OUT=p.REPO/'data/figures/pathway-selection-theory'

def log_mfpt(temp,absorber=0,wall=np.inf):
    """Exact B=0 backward-diffusion quadrature, start at +L=d/2.

Analytical inner integral uses Dawson's function to avoid low-T overflow.
"""
    L=5.;beta=1/(p.KB*temp);b=beta*p.HEIGHT;root=np.sqrt(b)
    tail=np.sqrt(np.pi/(2*beta*p.STIFFNESS))
    if np.isfinite(wall):tail*=erf((wall-L)*np.sqrt(beta*p.STIFFNESS/2))
    def integral(s):
        inner=L/root*(dawsn(root)-np.exp(b*((s/L)**2-1))*dawsn(root*s/L))+tail
        return np.exp(-b*(s/L)**2)*inner
    val,error=quad(integral,absorber,L,epsabs=1e-12,epsrel=1e-10,points=[0] if absorber<0 else None)
    return b+np.log(val/(p.KB*temp/h.GAMMA)),error

def bare_survival(dx=.005):
    """Absorbing hill, reflecting right wall, B=0; exact matrix exponential."""
    s=np.arange(0,5.15+dx/2,dx);u=p.trap_potential(s);f,b=v.rates(u,300,dx)
    diag=-(np.r_[f,0]+np.r_[0,b])[1:]
    off=np.sqrt(f[1:]*b[1:]);lam,U=eigh_tridiagonal(diag,off)
    weight=np.exp(-u[1:]/(2*p.KB*300));j=np.argmin(abs(s[1:]-5))
    coefficients=(weight@U)*U[j]/weight[j]
    times=np.unique(np.r_[np.linspace(0,30,301),60,143.178905])
    survival=np.exp(times[:,None]*lam)@coefficients
    assert abs(survival[0]-1)<1e-10 and np.all(np.diff(survival)<=1e-10)
    survival=np.clip(survival,0,1)
    frame=pd.DataFrame({'t_s':times,'survival_single_particle':survival,
                        'p_first_hill_single':1-survival,'p_first_hill_among_100':1-survival**100,'dx_um':dx})
    # Mean FPT solves the backward birth/death equation, including absorption.
    up=np.r_[f[1:],0];down=b
    band=np.zeros((3,len(diag)));band[1]=diag;band[0,1:]=up[:-1];band[2,:-1]=down[1:]
    mfpt=solve_banded((1,1),band,-np.ones(len(diag)))[j]
    return frame,float(mfpt)

def baseline():
    rows=[]
    for temp in [300,.1]:
        for absorb in [0,-5]:
            logt,err=log_mfpt(temp,absorb)
            rows.append({'temperature_K':temp,'absorber_um':absorb,'initial_um':5,'barrier_over_kBT':p.HEIGHT/(p.KB*temp),
                         'log10_mean_first_passage_s':logt/np.log(10),'mean_first_passage_s':np.exp(logt) if logt<700 else np.nan,
                         'quadrature_absolute_error_scaled_integral':err,'right_wall':'natural confining boundary at infinity'})
    pd.DataFrame(rows).to_csv(OUT/'bare_trap_mfpt.csv',index=False)
    checks=[];traces=[]
    for dx in [.01,.005]:
        frame,mfpt=bare_survival(dx);traces.append(frame)
        exact=np.exp(log_mfpt(300,0,5.15)[0])
        checks.append({'dx_um':dx,'discrete_mean_first_passage_s':mfpt,'quadrature_mean_first_passage_s':exact,'relative_error':abs(mfpt/exact-1)})
    assert checks[-1]['relative_error']<1e-4
    pd.concat(traces).to_csv(OUT/'bare_trap_survival.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'bare_trap_numerical_checks.csv',index=False)
    print('Bare baseline:',rows,flush=True);print('Survival checks:',checks,flush=True)

@njit(cache=True)
def continuation(initial,tstart,tau,dt,indptr,source,rx,ry,ex,ey,ax,ay,mult,is_h,group_size):
    s=initial.copy();tend=tau;ns=int(np.ceil((tend-tstart)/dt));step=(tend-tstart)/ns
    first=np.nan;peak=0;minv=1e9
    for k in range(ns):
        t=tstart+(k+.5)*step;w=np.sin(np.pi*t/(2*tau))**2
        force=n.magnetic(s,w,indptr,source,rx,ry,ex,ey,ax,ay,mult)
        crossed=0
        for i in range(len(s)):
            s[i]=n.trap_resolvent(s[i]+step*force[i]/h.GAMMA,step)
            if not is_h[i]:
                minv=min(minv,s[i])
                if s[i]<0:crossed+=1
        if crossed>peak:peak=crossed
        if crossed and np.isnan(first):first=tstart+(k+1)*step
    final_v=0;final_h=0
    for i in range(len(s)):
        if s[i]<0:
            if is_h[i]:final_h+=1
            else:final_v+=1
    return final_v,final_h,first,peak,minv,s

class ModeTest:
    def __init__(self,tau=8,steps=900):
        self.tau=tau;self.model=c.CoupledModel();self.lattice=a.Lattice(self.model);self.sol=self.model.solve(tau);self.reduced=n.Reduction(1,5)
        old=pd.read_csv(a.OUT/'summary-n3600.csv').query('tau_s==@tau').iloc[0]
        self.theta_start=old.unstable_window_start_deg;self.theta_end=old.unstable_window_end_deg
        grid=np.linspace(self.theta_start,self.theta_end,steps+1)*np.pi/180
        G=np.broadcast_to(np.eye(8),(25,8,8)).astype(complex).copy()
        # Propagator across the mechanically identified first unstable interval.
        for left,right in zip(grid[:-1],grid[1:]):
            theta=(left+right)/2;t=theta*tau/(np.pi/2);A=self.lattice.blocks(self.sol.sol(t),theta)
            eig,V=np.linalg.eigh(A);M=np.einsum('kai,ki,kbi->kab',V,np.exp(eig*(right-left)*tau/(np.pi/2)),V.conj());G=M@G
        U,s,Vh=np.linalg.svd(G);kk=np.argmax(s[:,0]);self.kk=int(kk);self.gain=float(s[kk,0])
        self.vec=Vh[kk,0].conj();self.vec*=np.exp(-1j*np.angle(self.vec[np.argmax(abs(self.vec))]))
        assert self.lattice.wave[kk,0]==0,'40-coordinate continuation must retain chosen wavevector'
        (OUT/'mode_selection.json').write_text(json.dumps({'tau_s':tau,'theta_start_deg':self.theta_start,'theta_end_deg':self.theta_end,
          'wavevector_indices':self.lattice.wave[kk].tolist(),'max_singular_gain':self.gain,'selection':'right singular vector of full time-ordered propagator; no empirical frequency fit',
          'complex_cell_vector_real':self.vec.real.tolist(),'complex_cell_vector_imag':self.vec.imag.tolist()},indent=2))
    def direction(self,phase=0):
        mode=np.empty(200)
        for cc,sites in enumerate(self.lattice.order):mode[sites]=(np.exp(1j*phase)*self.lattice.phase[self.kk,cc]*self.vec).real
        mode/=np.max(abs(mode[~self.model.is_h]))
        assert np.allclose(mode,mode[self.reduced.ref][self.reduced.mapping])
        return mode[self.reduced.ref]
    def shoot(self,amplitude,theta,phase,dt):
        tstart=theta*self.tau/90;q=self.sol.sol(tstart);base=np.where(self.reduced.is_h,*q)
        direction=self.direction(phase);initial=base+amplitude*direction
        assert np.all(initial[~self.reduced.is_h]>0),'initial perturbation already crossed hill'
        result=continuation(initial,tstart,self.tau,dt,*self.reduced.args())
        nv,nh,first,peak,minv,final=result
        return {'tau_s':self.tau,'theta_seed_deg':theta,'phase_rad':phase,'initial_amplitude_um':amplitude,'dt_s':dt,
                'final_vertical_crossed_groups':int(nv),'final_horizontal_crossed_groups':int(nh),
                'first_vertical_angle_deg':first*90/self.tau if np.isfinite(first) else np.nan,
                'peak_vertical_crossed_groups':int(peak),'minimum_vertical_coordinate_um':float(minv),
                'physical_vertical_defects':int(nv)*self.reduced.group_size,'initial_vertical_min_um':float(initial[~self.reduced.is_h].min())}
    def thresholds(self,dt=.0025,theta=None,phases=(0,np.pi/10)):
        theta=self.theta_start if theta is None else theta
        rows=[];brackets=[]
        for phase in phases:
            scan=[0,.05,.1,.2,.4,.8,1.6,3.2]
            # Ensure every trial starts with all verticals in the original well.
            q=self.sol.sol(theta*self.tau/90);limit=.99*q[1]
            scan=[x for x in scan if x<limit]
            previous=None;found=None
            for amp in scan:
                row=self.shoot(amp,theta,phase,dt);rows.append(row)
                if row['final_vertical_crossed_groups']:
                    found=(previous,row);break
                previous=row
            if found is None:
                brackets.append({'tau_s':self.tau,'theta_seed_deg':theta,'phase_rad':phase,'dt_s':dt,'status':'no persistent switch found before initially crossing perturbation limit'});continue
            lo,hi=found
            for _ in range(14):
                mid=(lo['initial_amplitude_um']+hi['initial_amplitude_um'])/2
                row=self.shoot(mid,theta,phase,dt);rows.append(row)
                if row['final_vertical_crossed_groups']:hi=row
                else:lo=row
            brackets.append({'tau_s':self.tau,'theta_seed_deg':theta,'phase_rad':phase,'dt_s':dt,
                'amplitude_lower_no_switch_um':lo['initial_amplitude_um'],'amplitude_upper_switch_um':hi['initial_amplitude_um'],
                'physical_vertical_defects_at_upper':hi['physical_vertical_defects'],'first_vertical_angle_deg_at_upper':hi['first_vertical_angle_deg'],
                'status':'direction-specific deterministic switch bracket; not a stochastic transition estimate'})
        return pd.DataFrame(rows),pd.DataFrame(brackets)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['baseline','shoot'],default='baseline');parser.add_argument('--dt',type=float,default=.0025);parser.add_argument('--theta',type=float)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='baseline':baseline()
    else:
        test=ModeTest();trials,brackets=test.thresholds(args.dt,args.theta)
        tag=f'theta{test.theta_start if args.theta is None else args.theta:g}-dt{args.dt:g}'
        trials.to_csv(OUT/f'deterministic_trials-{tag}.csv',index=False);brackets.to_csv(OUT/f'deterministic_brackets-{tag}.csv',index=False)
        print('Mode gain:',test.gain,'wavevector:',test.lattice.wave[test.kk],flush=True);print(brackets.to_string(index=False),flush=True)
