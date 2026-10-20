"""Along-trap linear fluctuations about the audited coherent H/V trajectory.

25 Fourier blocks retain all 200 site coordinates, including nonuniform modes.
This Gaussian diagnostic is not a nonlinear escape probability.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import brentq
import coupled_relaxation_analysis as c
import collective_horizontal_analysis as h
import rotating_potential_analysis as p
OUT=p.REPO/'data/figures/collective-stability'

def pair_gradient(dr,w):
    r2=np.sum(dr**2,axis=-1)
    inv=np.where((r2>0)&(r2<c.RC**2),1/np.maximum(r2,1e-30),0)
    r5=inv**2.5;x=dr[...,0]
    a=1-5*w*x*x*inv
    aa=np.stack([a+2*w,a],axis=-1)
    da=10*w*x[...,None]**2*dr*inv[...,None]**2
    da[...,0]-=10*w*x*inv
    eye=np.eye(2)
    return 3*c.COEFFICIENT_MD*r5[...,None,None]*(eye*aa[..., :,None]-5*dr[..., :,None]*dr[...,None,:]*inv[...,None,None]*aa[..., :,None]+dr[..., :,None]*da[...,None,:])

class Lattice:
    def __init__(self,model):
        self.model=model
        xy=model.centers[:,:2]
        labels,kind=np.unique(np.round(xy%60,8),axis=0,return_inverse=True)
        cell=np.floor(xy/60+1e-8).astype(int)
        self.cells=np.array([(x,y) for x in range(5) for y in range(5)])
        self.order=np.array([[np.flatnonzero((cell==cc).all(1)&(kind==a))[0] for a in range(8)] for cc in self.cells])
        self.ref=self.order[0];self.axis=model.axes[:,:2]
        assert np.allclose(self.axis[self.order],self.axis[self.ref][None])
        self.wave=self.cells.copy()
        self.phase=np.exp(2j*np.pi*(self.wave@self.cells.T)/5)
        self.hmask=model.is_h[self.ref]
        self.coherent_basis=np.column_stack([self.hmask,~self.hmask]).astype(float)/2
    def rows(self,q,theta):
        pos=self.model.positions(q)[:,:2]
        rows=np.zeros((8,200));selfdiag=np.zeros(8)
        dr0=pos[self.ref,None,:]-pos[None,:,:]
        dr0-=300*np.rint(dr0/300)
        for x in [-300,0,300]:
            for y in [-300,0,300]:
                dr=dr0+[x,y];K=pair_gradient(dr,np.sin(theta)**2)
                rows-=np.einsum('ia,ijab,jb->ij',self.axis[self.ref],K,self.axis)
                selfdiag+=np.einsum('ia,ijab,ib->i',self.axis[self.ref],K,self.axis[self.ref])
        rows[np.arange(8),self.ref]+=selfdiag
        ss=np.where(self.model.is_h,q[0],q[1])
        rows[np.arange(8),self.ref]+=np.where(abs(ss[self.ref])<5,8*p.HEIGHT/100,-p.STIFFNESS)
        return rows
    def blocks(self,q,theta):
        row=self.rows(q,theta)
        blocks=np.einsum('kc,acb->kab',self.phase,row[:,self.order])/h.GAMMA
        assert np.max(abs(blocks-blocks.conj().transpose(0,2,1)))<1e-7
        return .5*(blocks+blocks.conj().transpose(0,2,1))
    def force(self,s,theta):
        pos=self.model.centers[:,:2]+s[:,None]*self.axis
        dr0=pos[:,None]-pos[None,:];dr0-=300*np.rint(dr0/300)
        force=np.zeros((200,3))
        for x in [-300,0,300]:
            for y in [-300,0,300]:
                dr=np.concatenate([dr0+[x,y],np.zeros((200,200,1))],axis=-1)
                force+=c.magnetic_force(dr,np.sin(theta)**2).sum(axis=1)
        return (force[:,:2]*self.axis).sum(axis=1)+c.trap_force(s)
    def verify(self):
        q=np.array([2.3,4.2]);theta=np.deg2rad(47)
        ss=np.where(self.model.is_h,*q)
        rows=self.rows(q,theta);eps=1e-5
        columns=[0,1,10,100,101,110,199]
        err=[]
        for j in columns:
            d=np.zeros(200);d[j]=eps
            diff=(self.force(ss+d,theta)-self.force(ss-d,theta))/(2*eps)
            err.append(np.max(abs(diff[self.ref]-rows[:,j])))
        assert max(err)<1e-8,err
        # Reconstruct the real matrix and compare its spectrum with all blocks.
        full=np.empty((200,200))
        for ci,cc in enumerate(self.cells):
            translated=(self.cells-cc)%5
            order=np.array([np.flatnonzero((self.cells==t).all(1))[0] for t in translated])
            full[np.ix_(self.order[ci],self.order.ravel())]=rows[:,self.order[order]].reshape(8,200)
        errsym=np.max(abs(full-full.T))
        eigerr=np.max(abs(np.linalg.eigvalsh(full/h.GAMMA)-np.sort(np.linalg.eigvalsh(self.blocks(q,theta)).ravel())))
        assert errsym<1e-8 and eigerr<1e-7,(errsym,eigerr)
        # One constant-J noise step must agree with the unreduced 200-site system.
        lam,U=np.linalg.eigh(full/h.GAMMA);dt=.017
        noise=np.expm1(2*lam*dt)/(2*lam)
        densevar=(abs(U)**2)@noise
        lb,Ub=np.linalg.eigh(self.blocks(q,theta))
        nb=np.expm1(2*lb*dt)/(2*lb)
        blockvar=np.einsum('kai,ki->ka',abs(Ub)**2,nb).mean(axis=0)
        coverr=np.max(abs(densevar[self.ref]-blockvar))
        assert coverr<1e-10,coverr
        return {'finite_difference_max_pN_per_um':max(err),'matrix_symmetry_error':errsym,'Fourier_spectrum_error_per_s':eigerr,'constant_J_noise_variance_reduction_error':coverr}

def run(lattice,tau,nstep):
    sol=lattice.model.solve(tau)
    C=np.zeros((25,8,8),complex);Cd=np.zeros(8)
    G=np.broadcast_to(np.eye(8),(25,8,8)).astype(complex).copy();growth_started=False;growth_finished=False;growth_start=None;growth_end=None
    D=p.KB*300/h.GAMMA
    # Split steps at background trap joins, rather than averaging a stiff wall
    # and the shallow interior within one exponential step.
    grid=np.linspace(0,tau,nstep+1);joins=[]
    scan=np.linspace(0,tau,4001)
    for axis in range(2):
        for wall in [-5,5]:
            values=sol.sol(scan)[axis]-wall
            for i in np.flatnonzero(values[:-1]*values[1:]<0):
                joins.append(brentq(lambda t:sol.sol(t)[axis]-wall,scan[i],scan[i+1],xtol=1e-12))
    grid=np.unique(np.r_[grid,joins]);rows=[]
    for k,(left,right) in enumerate(zip(grid[:-1],grid[1:])):
        dt=right-left;t=(left+right)/2;q=sol.sol(t);A=lattice.blocks(q,np.pi*t/(2*tau))
        lam,V=np.linalg.eigh(A)
        E=np.exp(lam*dt)
        M=np.einsum('kai,ki,kbi->kab',V,E,V.conj())
        xx=2*lam*dt
        noise=np.empty_like(lam);small=abs(xx)<1e-7
        noise[small]=dt*(1+xx[small]/2+xx[small]**2/6)
        noise[~small]=np.expm1(xx[~small])/(2*lam[~small])
        Q=np.einsum('kai,ki,kbi->kab',V,2*D*noise,V.conj())
        C=M@C@M.conj().transpose(0,2,1)+Q
        # Fixed, reproducible window: first interval of positive maximum eigenvalue.
        if not growth_started and lam.max()>0:
            growth_started=True;growth_start=left
        if growth_started and not growth_finished:
            if lam.max()<=0:
                growth_finished=True;growth_end=left
            else:G=M@G
        diagonal=A.mean(axis=0).diagonal().real
        ex=np.exp(2*diagonal*dt)
        nd=np.where(abs(diagonal)>1e-10,np.expm1(2*diagonal*dt)/(2*np.where(abs(diagonal)>1e-10,diagonal,1)),dt)
        Cd=ex*Cd+2*D*nd
        variances=C.diagonal(axis1=1,axis2=2).real.mean(axis=0)
        sv=np.sqrt(np.maximum(variances,0));tt=right;qq=sol.sol(tt)
        top=np.unravel_index(np.argmax(lam),lam.shape);mode=V[top[0],:,top[1]]
        idx=lattice.hmask;v=~idx
        # Tail at the current time; neither first passage nor retained occupancy.
        row={'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'t_s':tt,'theta_deg':90*tt/tau,'s_h_um':qq[0],'s_v_um':qq[1],
             'sigma_h_300K_um':np.sqrt(variances[idx].mean()),'sigma_v_300K_um':np.sqrt(variances[v].mean()),
             'sigma_v_diagonal_300K_um':np.sqrt(Cd[v].mean()),
             'gaussian_vertical_tail_300K':ndtr(-qq[1]/np.maximum(sv[v],1e-30)).mean(),
             'gaussian_vertical_outer_branch_300K':ndtr((qq[1]-5)/np.maximum(sv[v],1e-30)).mean(),
             'k0_fraction_vertical_variance':float(C[0].diagonal().real[v].sum()/np.maximum(C.diagonal(axis1=1,axis2=2).real[:,v].sum(),1e-30)),
             'gaussian_vertical_tail_0p1K':ndtr(-qq[1]/np.maximum(sv[v]*np.sqrt(.1/300),1e-30)).mean(),
             'largest_eigenvalue_per_s':lam.max(),'largest_k0_eigenvalue_per_s':lam[0].max(),
             'largest_coherent_HV_eigenvalue_per_s':np.linalg.eigvalsh(lattice.coherent_basis.T@A[0]@lattice.coherent_basis).max(),
             'dominant_kx':lattice.wave[top[0],0],'dominant_ky':lattice.wave[top[0],1],
             'dominant_vertical_weight':float(np.sum(abs(mode[v])**2)),
             'vertical_sigma_over_hill_distance_300K':np.max(sv[v])/qq[1],
             'horizontal_sigma_over_hill_distance_300K':np.max(sv[idx])/max(abs(qq[0]),1e-12)}
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/f'trace-tau{tau:g}-n{nstep}.csv',index=False)
    peak=frame.iloc[frame.gaussian_vertical_tail_300K.argmax()]
    summary={'tau_s':tau,'nstep':nstep,'omega_rad_s':np.pi/(2*tau),'peak_gaussian_tail_300K':peak.gaussian_vertical_tail_300K,
             'peak_tail_angle_deg':peak.theta_deg,'sigma_v_at_peak_um':peak.sigma_v_300K_um,
             'sigma_v_diagonal_at_peak_um':peak.sigma_v_diagonal_300K_um,
             'peak_gaussian_tail_0p1K':frame.gaussian_vertical_tail_0p1K.max(),
             'max_eigenvalue_per_s':frame.largest_eigenvalue_per_s.max(),
             'max_k0_eigenvalue_per_s':frame.largest_k0_eigenvalue_per_s.max(),
             'max_vertical_sigma_over_hill_distance_300K':frame.vertical_sigma_over_hill_distance_300K.max(),
             'max_horizontal_sigma_um':frame.sigma_h_300K_um.max(),
             'unstable_window_start_deg':90*growth_start/tau if growth_started else np.nan,
             'unstable_window_end_deg':90*(growth_end if growth_finished else tau)/tau if growth_started else np.nan,
             'unstable_window_max_singular_gain':float(np.linalg.svd(G,compute_uv=False).max()),
             'unstable_window_k0_singular_gain':float(np.linalg.svd(G[0],compute_uv=False).max()),
             'outer_branch_probability_at_peak_tail_300K':peak.gaussian_vertical_outer_branch_300K,
             'k0_fraction_vertical_variance_at_peak':peak.k0_fraction_vertical_variance,
             'status':'linear Gaussian diagnostic; nonlinear crossing not validated'}
    print(summary,flush=True)
    return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--taus',nargs='+',type=float,default=[7,8,9,15,30,40]);parser.add_argument('--nstep',type=int,default=900)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    model=c.CoupledModel();lattice=Lattice(model)
    checks=lattice.verify();print(checks,flush=True)
    (OUT/'jacobian_checks.json').write_text(json.dumps(checks,indent=2))
    rows=[run(lattice,tau,args.nstep) for tau in args.taus]
    target=OUT/f'summary-n{args.nstep}.csv'
    old=pd.read_csv(target) if target.exists() else pd.DataFrame()
    pd.concat([old,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(['tau_s','nstep'],keep='last').sort_values('tau_s').to_csv(target,index=False)
