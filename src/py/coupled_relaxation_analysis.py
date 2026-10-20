"""Coherent H/V relaxation with magcolloids' planar finite-cutoff dipolar forces.

All in-plane periodic images inside 200 um; static LAMMPS force checks only.
No time-stepped MD runs. Previous figures and probability tables are untouched.
"""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.optimize import brentq
import rotating_potential_analysis as p
import collective_horizontal_analysis as h
import vertical_competition_analysis as v
OUT=p.REPO/'data/figures/coupled-relaxation'
RC=200.
# magcolloids converts B using 2.99e8 m/s; LAMMPS' electrostatic conversion
# corresponds to 299792458 m/s. This is a unit convention, not a fitted factor.
COEFFICIENT_MD=p.COEFFICIENT*(299792458/2.99e8)**2

def trap_force(s):
    return np.where(np.abs(s)<5,8*p.HEIGHT*np.asarray(s)/100,-p.STIFFNESS*(np.asarray(s)-5*np.sign(s)))

def magnetic_force(dr,w):
    r2=np.sum(dr*dr,axis=-1)
    inv=np.where((r2<RC**2)&(r2>0),1/np.maximum(r2,1e-30),0)
    r5=inv**2.5
    force=3*COEFFICIENT_MD*r5[...,None]*dr*(1-5*w*dr[...,0,None]**2*inv[...,None])
    force[...,0]+=6*COEFFICIENT_MD*w*dr[...,0]*r5
    return force

class CoupledModel:
    def __init__(self,nq=161):
        self.state=pd.read_csv(p.STATE)
        self.centers=self.state[['x','y','z']].to_numpy()
        rel=self.state[['cx','cy','cz']].to_numpy()
        self.is_h=self.state.dx.abs().to_numpy()>.5
        self.axes=rel/5
        self.uh=self.axes*self.is_h[:,None];self.uv=self.axes*(~self.is_h)[:,None]
        i,j=np.triu_indices(len(rel),1)
        r0=self.centers[i]-self.centers[j];r0-=p.BOX*np.rint(r0/p.BOX)
        a=self.uh[i]-self.uh[j];b=self.uv[i]-self.uv[j]
        rows=[]
        for x in [-300,0,300]:
            for y in [-300,0,300]:
                r=r0+[x,y,0]
                keep=np.linalg.norm(r,axis=1)<RC+21
                rows.append(np.column_stack([r[keep],a[keep],b[keep]]))
        unique,count=np.unique(np.vstack(rows),axis=0,return_counts=True)
        self.r0=unique[:,:3];self.a=unique[:,3:6];self.b=unique[:,6:9];self.count=count
        q=np.linspace(-5.1,5.1,nq)
        values=np.empty((nq,nq,2,2))
        for i,sh in enumerate(q):
            for j,sv in enumerate(q):values[i,j]=self.basis(sh,sv)
        self.splines=[[RectBivariateSpline(q,q,values[:,:,k,w]) for w in range(2)] for k in range(2)]
    def basis(self,sh,sv):
        dr=self.r0+sh*self.a+sv*self.b
        f0=magnetic_force(dr,0);f1=magnetic_force(dr,1)-f0
        return np.array([[(f*self.a*self.count[:,None]).sum()/100 for f in [f0,f1]],
                         [(f*self.b*self.count[:,None]).sum()/100 for f in [f0,f1]]])
    def force(self,q,theta):
        w=np.sin(theta)**2
        return np.array([s[0].ev(q[0],q[1])+w*s[1].ev(q[0],q[1]) for s in self.splines])+trap_force(q)
    def positions(self,q):return self.centers+q[0]*self.uh+q[1]*self.uv
    def solve(self,tau,frozen=False,tol=1e-8):
        def rhs(t,q):
            f=self.force(q,np.pi*t/(2*tau))/h.GAMMA
            if frozen:f[1]=0
            return f
        def cross(t,q):return q[0]
        cross.direction=-1
        sol=solve_ivp(rhs,[0,tau],[5,5],method='Radau',rtol=tol,atol=tol/10,max_step=tau/300,dense_output=True,events=cross)
        assert sol.success,sol.message
        assert np.max(np.abs(sol.y))<5.1
        return sol
    def tagged_force(self,s,q,theta):
        j=np.flatnonzero(~self.is_h)[0];pos=self.positions(q)
        ss=np.atleast_1d(s)
        ghosts=[]
        for x in [-300,0,300]:
            for y in [-300,0,300]:
                image=pos+[x,y,0]
                keep=np.linalg.norm(image-self.centers[j],axis=1)<RC+np.max(abs(ss))
                keep[j]=False
                ghosts.append(image[keep])
        ghosts=np.vstack(ghosts)
        # The tagged site is vertical and planar. Compute the projected force
        # directly instead of constructing three large Cartesian force arrays.
        dx=self.centers[j,0]-ghosts[:,0]
        dy=self.centers[j,1]+ss[:,None]*self.axes[j,1]-ghosts[None,:,1]
        r2=dx[None,:]**2+dy**2
        inv=np.where(r2<RC**2,1/r2,0)
        w=np.sin(theta)**2
        force=(3*COEFFICIENT_MD*dy*inv**2.5*(1-5*w*dx[None,:]**2*inv)).sum(axis=1)*self.axes[j,1]
        return force+trap_force(ss)

class PathPotential:
    def __init__(self,model,sol,tau,dx=.005,nt=901):
        from scipy.integrate import cumulative_trapezoid
        self.dx=dx;self.s=np.arange(-5.2,5.2+dx/2,dx)
        times=np.linspace(0,tau,nt)
        # Integrate the actual cut-off force. A raw truncated pair energy has
        # jumps at rc; differentiating those jumps would add fictitious impulses.
        profiles=[]
        for t in times:
            force=model.tagged_force(self.s,sol.sol(t),np.pi*t/(2*tau))
            profiles.append(-cumulative_trapezoid(force,self.s,initial=0))
        self.times=times;self.u=np.array(profiles);self.tau=tau
    def energy_at(self,t):
        x=t/self.tau*(len(self.times)-1);i=min(int(x),len(self.times)-2);f=x-i
        return (1-f)*self.u[i]+f*self.u[i+1]

def probability(pot,tau,temp=300,dt=.005):
    mass=np.zeros(len(pot.s));mass[np.argmin(abs(pot.s-5))]=1
    n=int(np.ceil((tau-.05)/dt));step=(tau-.05)/n;band=np.zeros((3,len(mass)))
    rows=[];error=0.
    for k in range(n):
        t=(k+1)*step;f,b=v.rates(pot.energy_at(t),temp,pot.dx)
        band[1]=1;band[1,:-1]+=step*f;band[1,1:]+=step*b
        band[0,1:]=-step*b;band[2,:-1]=-step*f
        mass=solve_banded((1,1),band,mass,check_finite=False)
        error=max(error,abs(mass.sum()-1))
        if k%10==0 or k==n-1:
            pv=mass[pot.s<-pot.dx/4].sum()+.5*mass[abs(pot.s)<pot.dx/4].sum()
            rows.append([t,t*90/tau,pv,mass@pot.s])
    assert mass.min()>=-1e-13 and error<1e-8
    return pd.DataFrame(rows,columns=['t_s','theta_deg','p_vertical','mean_s_v_um']),{'mass_error':error,'outer_wall_mass':float(mass[abs(pot.s)>5.15].sum())}

def compute(taus,dx=.005,dt=.005,nt=901):
    OUT.mkdir(parents=True,exist_ok=True);model=CoupledModel()
    print('Geometry grouped into',len(model.count),'unique interactions',flush=True)
    rows=[];paths=[];traces=[]
    for tau in taus:
        for frozen in [True,False]:
            label='fixed verticals' if frozen else 'coupled relaxation'
            sol=model.solve(tau,frozen)
            t=np.arange(int(round(tau*20)))*.05;q=sol.sol(t)
            crossed=np.flatnonzero(q[0]<=0)
            vertical_crossed=np.any(q[1]<=0)
            path=pd.DataFrame({'tau_s':tau,'model':label,'t_s':t,'theta_deg':90*t/tau,'s_h_um':q[0],'s_v_um':q[1]});paths.append(path)
            row={'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'model':label,
                 'horizontal_crossing_angle_deg':90*sol.t_events[0][0]/tau if len(sol.t_events[0]) else np.nan,
                 'horizontal_first_crossed_frame_angle_deg':90*t[crossed[0]]/tau if len(crossed) else np.nan,
                 'vertical_min_um':q[1].min(),'vertical_background_crossed':bool(vertical_crossed)}
            if vertical_crossed:
                row['p_vertical_final']=np.nan
                row['status']='coherent vertical background crosses; rare-defect approximation invalid'
            else:
                pot=PathPotential(model,sol,tau,dx,nt)
                trace,checks=probability(pot,tau,dt=dt)
                trace['tau_s']=tau;trace['model']=label;traces.append(trace)
                row.update({'p_vertical_final':trace.p_vertical.iloc[-1],'status':'non-switching background',**checks})
            rows.append(row);print(row,flush=True)
    suffix=f'dx{dx:g}-dt{dt:g}-nt{nt}'
    pd.DataFrame(rows).to_csv(OUT/f'summary-{suffix}.csv',index=False)
    pd.concat(paths).to_csv(OUT/f'paths-{suffix}.csv',index=False)
    if traces:pd.concat(traces).to_csv(OUT/f'probabilities-{suffix}.csv',index=False)
    return model,pd.DataFrame(rows)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--taus',nargs='+',type=float,default=[7,8,9,10,15]);parser.add_argument('--dx',type=float,default=.005);parser.add_argument('--dt',type=float,default=.005);parser.add_argument('--nt',type=int,default=901)
    a=parser.parse_args();compute(a.taus,a.dx,a.dt,a.nt)
