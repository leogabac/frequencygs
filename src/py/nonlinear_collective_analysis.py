"""Reduced along-trap Langevin model with nonlinear dipoles and trap branches.

No LAMMPS trajectories; optional cell restrictions are exact symmetry subspaces.
Noise is projected by the same restriction, not tuned to a transition.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/frequencygs-numba')
import json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit,prange,set_num_threads
import coupled_relaxation_analysis as c
import collective_horizontal_analysis as h
import rotating_potential_analysis as p
OUT=p.REPO/'data/figures/nonlinear-collective'

@njit(cache=True)
def magnetic(s,w,indptr,source,rx,ry,ex,ey,ax,ay,mult):
    force=np.zeros(len(s))
    for i in range(len(s)):
        for l in range(indptr[i],indptr[i+1]):
            j=source[l];x=rx[l]+ax[i]*s[i]-ex[l]*s[j];y=ry[l]+ay[i]*s[i]-ey[l]*s[j]
            r2=x*x+y*y
            if r2<40000 and r2>0:
                inv=1/r2;fac=3*c.COEFFICIENT_MD*inv*inv*np.sqrt(inv)
                angular=1-5*w*x*x*inv
                force[i]+=mult[l]*fac*(ax[i]*x*(angular+2*w)+ay[i]*y*angular)
    return force

@njit(cache=True)
def trap_resolvent(y,dt):
    # Implicit trap drift, including the cusp at the shallow/stiff join.
    a=8*p.HEIGHT/100/h.GAMMA;b=p.STIFFNESS/h.GAMMA
    z=abs(y);sign=1 if y>=0 else -1
    if z<5*(1-a*dt):return y/(1-a*dt)
    if z<=5:return 5*sign
    return sign*(z+5*b*dt)/(1+b*dt)

@njit(cache=True,parallel=True)
def ensemble(tau,temp,dt,nrun,seed,indptr,source,rx,ry,ex,ey,ax,ay,mult,is_h,group_size):
    # One final observation time matches the saved 20 Hz data (tau - .05 s).
    tend=tau-.05;nstep=int(np.ceil(tend/dt));step=tend/nstep
    stride=max(1,int(round(.05/step)));nframe=nstep//stride+1+(nstep%stride!=0);nsite=len(ax)
    trace=np.zeros((nrun,nframe,5));final=np.empty((nrun,nsite))
    noise=np.sqrt(2*p.KB*temp/h.GAMMA/group_size*step)
    for run in prange(nrun):
        np.random.seed(seed+run)
        s=np.full(nsite,5.);ever=np.zeros(nsite,np.bool_);frame=0
        trace[run,frame,0]=0;frame+=1
        for k in range(nstep):
            t=(k+.5)*step;w=np.sin(np.pi*t/(2*tau))**2
            f=magnetic(s,w,indptr,source,rx,ry,ex,ey,ax,ay,mult)
            for j in range(nsite):
                y=s[j]+step*f[j]/h.GAMMA+noise*np.random.normal()
                s[j]=trap_resolvent(y,step)
                if s[j]<0:ever[j]=True
            if (k+1)%stride==0 or k==nstep-1:
                nh=0;nv=0;ph=0.;pv=0.;ev=0.;rh=0.;rv=0.
                for j in range(nsite):
                    if is_h[j]:nh+=1;ph+=s[j]<0;rh+=abs(s[j]-5)/10
                    else:nv+=1;pv+=s[j]<0;ev+=ever[j];rv+=abs(s[j]-5)/10
                trace[run,frame,0]=(k+1)*step
                trace[run,frame,1]=pv/nv;trace[run,frame,2]=ev/nv
                trace[run,frame,3]=rh/nh;trace[run,frame,4]=rv/nv
                frame+=1
        final[run]=s
    return trace,final

class Reduction:
    def __init__(self,nx=1,ny=5):
        self.state=pd.read_csv(p.STATE);centers=self.state[['x','y']].to_numpy();axes=self.state[['cx','cy']].to_numpy()/5
        cell=np.floor(centers/60+1e-8).astype(int)
        _,kind=np.unique(np.round(centers%60,8),axis=0,return_inverse=True)
        label=np.column_stack([cell[:,0]%nx,cell[:,1]%ny,kind])
        _,self.mapping=np.unique(label,axis=0,return_inverse=True)
        self.ref=np.array([np.flatnonzero(self.mapping==j)[0] for j in range(self.mapping.max()+1)])
        self.nsite=len(self.ref);self.group_size=200//self.nsite;self.nx=nx;self.ny=ny
        self.ax=axes[self.ref,0];self.ay=axes[self.ref,1];self.is_h=abs(self.ax)>.5
        assert np.allclose(axes,axes[self.ref][self.mapping])
        # Group repeated interactions for a fixed representative. Keep a buffer
        # for the whole trap range; the actual radial cutoff is checked each step.
        data=[];indptr=[0]
        for ii,i in enumerate(self.ref):
            dr=centers[i]-centers;dr-=300*np.rint(dr/300);rows=[]
            for x in [-300,0,300]:
                for y in [-300,0,300]:
                    rr=dr+[x,y];keep=(np.sum(rr*rr,axis=1)<225**2)
                    keep[i]=False
                    rows.extend(np.column_stack([self.mapping[keep],rr[keep],axes[keep]]).tolist())
            unique,count=np.unique(rows,axis=0,return_counts=True)
            data.extend(np.column_stack([unique,count]).tolist());indptr.append(len(data))
        data=np.array(data);self.indptr=np.array(indptr,dtype=np.int64);self.source=data[:,0].astype(np.int64)
        self.rx=data[:,1];self.ry=data[:,2];self.ex=data[:,3];self.ey=data[:,4];self.mult=data[:,5]
    def args(self):return (self.indptr,self.source,self.rx,self.ry,self.ex,self.ey,self.ax,self.ay,self.mult,self.is_h,self.group_size)
    def verify(self):
        import collective_stability_analysis as a
        # Only geometry is needed by the reference force evaluator.
        class Geometry:pass
        geom=Geometry();geom.centers=self.state[['x','y','z']].to_numpy();geom.axes=self.state[['cx','cy','cz']].to_numpy()/5;geom.is_h=self.state.dx.abs().to_numpy()>.5
        lat=a.Lattice(geom)
        rng=np.random.default_rng(20261010);errors=[]
        for theta in [0,.8,1.4]:
            s=rng.uniform(-5.05,5.05,self.nsite)
            actual=lat.force(s[self.mapping],theta)[self.ref]-c.trap_force(s)
            fast=magnetic(s,np.sin(theta)**2,*self.args()[:-2])
            errors.append(float(np.max(abs(actual-fast))))
        assert max(errors)<1e-10,errors
        # The scalar resolvent obeys backward Euler on each smooth branch.
        dt=.005;err=[]
        for y in [-5.3,-4.,0.,4.,5.3]:
            z=trap_resolvent(y,dt);err.append(abs(z-y-dt*c.trap_force(z)/h.GAMMA))
        assert max(err)<1e-10,err
        return {'nonlinear_force_error_pN':errors,'trap_resolvent_residual_um':max(err),'nsite':self.nsite,'group_size':self.group_size,'directed_pair_terms':len(self.source)}

def compute(tau,temp,dt,nrun,seed,nx,ny):
    OUT.mkdir(parents=True,exist_ok=True);model=Reduction(nx,ny);checks=model.verify();print(checks,flush=True)
    start=time.monotonic();trace,final=ensemble(tau,temp,dt,nrun,seed,*model.args());elapsed=time.monotonic()-start
    tag=f'T{temp:g}-tau{tau:g}-sites{model.nsite}-dt{dt:g}-n{nrun}-seed{seed}'
    np.savez_compressed(OUT/f'ensemble-{tag}.npz',trace=trace,final=final,ref=model.ref,mapping=model.mapping)
    v=~model.is_h;perrun=(final[:,v]<0).mean(axis=1)
    stats={'temperature_K':temp,'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'nsite':model.nsite,'group_size':model.group_size,
           'nrun':nrun,'seed':seed,'dt_requested_s':dt,'t_obs_s':tau-.05,'p_vertical_final':perrun.mean(),
           'standard_error_over_independent_runs':perrun.std(ddof=1)/np.sqrt(nrun),
           'p_vertical_ever':trace[:,-1,2].mean(),'p_horizontal_final':(final[:,model.is_h]<0).mean(),
           'elapsed_s':elapsed,'status':'nonlinear along-trap model; time-step and mode-restriction checks required',
           'source':f'ensemble-{tag}.npz'}
    columns=['t_s','p_vertical','p_vertical_ever','Rh','Rv'];frame=pd.DataFrame(trace.mean(axis=0),columns=columns)
    frame['p_vertical_se']=trace[:,:,1].std(axis=0,ddof=1)/np.sqrt(nrun);frame['theta_deg']=90*frame.t_s/tau
    frame.to_csv(OUT/f'trace-{tag}.csv',index=False)
    (OUT/f'summary-{tag}.json').write_text(json.dumps({**stats,'checks':checks},indent=2))
    summaries=[{k:v for k,v in json.loads(f.read_text()).items() if k!='checks'} for f in sorted(OUT.glob('summary-*.json'))]
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    print(stats,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--tau',type=float,default=8);parser.add_argument('--temperature',type=float,default=300);parser.add_argument('--dt',type=float,default=.005)
    parser.add_argument('--nrun',type=int,default=64);parser.add_argument('--seed',type=int,default=61010);parser.add_argument('--nx',type=int,choices=[1,5],default=1);parser.add_argument('--ny',type=int,choices=[1,5],default=5);parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.tau<=.05 or args.dt<=0 or args.temperature<0 or args.nrun<2:parser.error('require tau > .05, dt > 0, temperature >= 0, nrun >= 2')
    set_num_threads(args.threads);compute(args.tau,args.temperature,args.dt,args.nrun,args.seed,args.nx,args.ny)
