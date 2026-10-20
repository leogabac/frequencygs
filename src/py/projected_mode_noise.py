"""Two real collective quadratures with projected nonlinear drift and noise.

A falsifiable two-coordinate closure, not an along-trap lattice ensemble.
The coherent reference trajectory is prescribed; orthogonal modes are omitted.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/frequencygs-numba')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit,prange,set_num_threads
import rotating_potential_analysis as p
import coupled_relaxation_analysis as c
import collective_horizontal_analysis as h
import collective_stability_analysis as a
import nonlinear_collective_analysis as n
from pathway_selection_theory import ModeTest,continuation
OUT=p.REPO/'data/figures/projected-mode-noise'

@njit(cache=True,parallel=True)
def tabulate(qs,ws,zgrid,E,indptr,source,rx,ry,ex,ey,ax,ay,mult,is_h,group_size):
    nz=len(zgrid);out=np.zeros((len(qs),nz,nz,2))
    for k in prange(len(qs)):
        base=np.where(is_h,qs[k,0],qs[k,1])
        for i in range(nz):
            for j in range(nz):
                s=base+E[:,0]*zgrid[i]+E[:,1]*zgrid[j]
                f=n.magnetic(s,ws[k],indptr,source,rx,ry,ex,ey,ax,ay,mult)
                out[k,i,j]=group_size*(E.T@f)/h.GAMMA
    return out

@njit(cache=True)
def interpolate(table,z0,z1,angle_index,zmin,dz):
    nk,nz=table.shape[:2];k=min(nk-2,int(angle_index));ft=angle_index-k
    x=(z0-zmin)/dz;y=(z1-zmin)/dz
    i=max(0,min(nz-2,int(np.floor(x))));j=max(0,min(nz-2,int(np.floor(y))))
    fx=x-i;fy=y-j
    out=np.empty(2)
    for d in range(2):
        a00=(1-ft)*table[k,i,j,d]+ft*table[k+1,i,j,d]
        a10=(1-ft)*table[k,i+1,j,d]+ft*table[k+1,i+1,j,d]
        a01=(1-ft)*table[k,i,j+1,d]+ft*table[k+1,i,j+1,d]
        a11=(1-ft)*table[k,i+1,j+1,d]+ft*table[k+1,i+1,j+1,d]
        out[d]=(1-fx)*(1-fy)*a00+fx*(1-fy)*a10+(1-fx)*fy*a01+fx*fy*a11
    return out

@njit(cache=True,parallel=True)
def noise_runs(tau,temp,dt,nrun,seed,table,q,E,is_h,group_size,zmin,dz):
    ns=int(np.ceil(tau/dt));step=tau/ns;nk=len(table);nq=len(q)
    stride=max(1,int(round(.05/step)));nf=ns//stride+1+(ns%stride!=0)
    trace=np.zeros((nrun,nf,5));final=np.zeros((nrun,2));outside=np.zeros(nrun,np.int64)
    noise=np.sqrt(2*p.KB*temp/h.GAMMA*step)
    for run in prange(nrun):
        np.random.seed(seed+run);z=np.zeros(2);ever=np.zeros(len(is_h),np.bool_);frame=1
        for k in range(ns):
            u=(k+.5)/ns;iq=min(nq-2,int(u*(nq-1)));fq=u*(nq-1)-iq
            qb=(1-fq)*q[iq]+fq*q[iq+1]
            f=interpolate(table,z[0],z[1],u*(nk-1),zmin,dz)
            for j in range(len(is_h)):
                base=qb[0] if is_h[j] else qb[1]
                s=base+E[j,0]*z[0]+E[j,1]*z[1]
                tf=8*p.HEIGHT*s/100 if abs(s)<5 else -p.STIFFNESS*(s-5*np.sign(s))
                f[0]+=group_size*E[j,0]*tf/h.GAMMA
                f[1]+=group_size*E[j,1]*tf/h.GAMMA
            z+=step*f+noise*np.random.normal(0,1,2)
            if max(abs(z[0]),abs(z[1]))>-zmin:outside[run]+=1
            u=(k+1)/ns;iq=min(nq-2,int(u*(nq-1)));fq=u*(nq-1)-iq
            qb=(1-fq)*q[iq]+fq*q[iq+1]
            pv=0.;nv=0
            for j in range(len(is_h)):
                if not is_h[j]:
                    s=qb[1]+E[j,0]*z[0]+E[j,1]*z[1]
                    ever[j]|=s<0;pv+=s<0;nv+=1
            if (k+1)%stride==0 or k==ns-1:
                trace[run,frame,0]=(k+1)*step;trace[run,frame,1]=pv/nv
                trace[run,frame,2]=ever[~is_h].sum()/nv
                trace[run,frame,3]=z[0];trace[run,frame,4]=z[1];frame+=1
        final[run]=z
    return trace,final,outside

def basis(test):
    full=np.empty((200,2))
    for cc,sites in enumerate(test.lattice.order):
        u=test.lattice.phase[test.kk,cc]*test.vec
        full[sites,0]=u.real;full[sites,1]=-u.imag
    full/=np.sqrt(25/2)
    assert np.allclose(full.T@full,np.eye(2),atol=1e-12)
    assert np.max(abs(full[test.model.is_h].sum(axis=0)))<1e-12
    assert np.max(abs(full[~test.model.is_h].sum(axis=0)))<1e-12
    E=full[test.reduced.ref]
    assert np.allclose(full,E[test.reduced.mapping])
    return full,E

def linear_projection(test,full,E,steps=3600):
    """Exact whole Fourier-block covariance, then its marginal in this pair.

Also integrate the actual two-quadrature Galerkin covariance. The distinction
tests omitted linear mixing before considering nonlinear escape.
"""
    C=np.zeros((8,8),complex);V=np.zeros((2,2));grid=np.linspace(0,test.tau,steps+1)
    # Split at every reference trap join, matching the audited linear solver.
    from scipy.optimize import brentq
    joins=[]
    for j in [0,1]:
        for lo,hi in zip(grid[:-1],grid[1:]):
            if (abs(test.sol.sol(lo)[j])-5)*(abs(test.sol.sol(hi)[j])-5)<0:
                joins.append(brentq(lambda t:abs(test.sol.sol(t)[j])-5,lo,hi))
    grid=np.unique(np.r_[grid,joins]);rows=[];D=p.KB*300/h.GAMMA
    # normalized unitary Fourier coefficient gives variance C; real quadratures
    # along the normalized conjugate pair each have variance vec^dagger C vec.
    for lo,hi in zip(grid[:-1],grid[1:]):
        t=(lo+hi)/2;theta=t/test.tau*np.pi/2
        A=test.lattice.blocks(test.sol.sol(t),theta)[test.kk]
        lam,U=np.linalg.eigh(A);dt=hi-lo;M=(U*np.exp(lam*dt))@U.conj().T
        ratio=np.where(abs(lam)>1e-10,np.expm1(2*lam*dt)/(2*lam),dt)
        C=M@C@M.conj().T+2*D*(U*ratio)@U.conj().T
        l=float(np.vdot(test.vec,A@test.vec).real)
        V=np.exp(2*l*dt)*V+2*D*(np.expm1(2*l*dt)/(2*l) if abs(l)>1e-10 else dt)*np.eye(2)
        var=float(np.vdot(test.vec,C@test.vec).real)
        rows.append({'t_s':hi,'theta_deg':90*hi/test.tau,'quadrature_variance_exact_block_um2':var,
          'quadrature_variance_two_coordinate_um2':V[0,0],'projected_growth_rate_s_inv':l})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'linear_noise_projection-tau8.csv',index=False)
    return frame

def prepare(nz=33,ntheta=181):
    test=ModeTest();full,E=basis(test);theta=np.linspace(0,np.pi/2,ntheta)
    q=test.sol.sol(theta*test.tau/(np.pi/2)).T;zs=np.linspace(-32,32,nz)
    start=time.monotonic();table=tabulate(q,np.sin(theta)**2,zs,E,*test.reduced.args())
    np.savez_compressed(OUT/f'drift-nz{nz}-ntheta{ntheta}.npz',table=table,E=E,zgrid=zs,theta=theta)
    rng=np.random.default_rng(20261011);err=[];exact=[]
    for _ in range(60):
        th=rng.uniform(0,np.pi/2);z=rng.uniform(-8,8,2);qb=test.sol.sol(th*test.tau/(np.pi/2))
        s=np.where(test.reduced.is_h,*qb)+E@z
        true=test.reduced.group_size*(E.T@n.magnetic(s,np.sin(th)**2,*test.reduced.args()[:-2]))/h.GAMMA
        interp=interpolate(table,*z,th/(np.pi/2)*(ntheta-1),zs[0],zs[1]-zs[0])
        err.append(np.linalg.norm(interp-true));exact.append(np.linalg.norm(true))
    checks={'basis_orthonormal_error':float(np.max(abs(full.T@full-np.eye(2)))),
      'noise_diffusivity_um2_per_s':p.KB*300/h.GAMMA,'n_real_coordinates':2,
      'magnetic_interpolation_rms_drift_error_um_per_s':float(np.sqrt(np.mean(np.array(err)**2))),
      'magnetic_exact_rms_drift_um_per_s':float(np.sqrt(np.mean(np.array(exact)**2))),
      'magnetic_interpolation_max_drift_error_um_per_s':float(max(err)),
      'elapsed_tabulation_s':time.monotonic()-start,'nz':nz,'ntheta':ntheta}
    (OUT/f'grid-checks-nz{nz}-ntheta{ntheta}.json').write_text(json.dumps(checks,indent=2))
    linear_projection(test,full,E)
    print(checks,flush=True)

def run(nz,ntheta,dt,nrun,seed):
    data=np.load(OUT/f'drift-nz{nz}-ntheta{ntheta}.npz');E=data['E'];zs=data['zgrid'];red=n.Reduction(1,5)
    model=c.CoupledModel();sol=model.solve(8);q=sol.sol(np.linspace(0,8,7201)).T
    start=time.monotonic();tr,z,outside=noise_runs(8,300,dt,nrun,seed,data['table'],q,E,red.is_h,red.group_size,zs[0],zs[1]-zs[0])
    assert outside.sum()==0,'Drift grid domain exceeded; enlarge and repeat'
    tag=f'nz{nz}-ntheta{ntheta}-dt{dt:g}-n{nrun}-seed{seed}'
    np.savez_compressed(OUT/f'runs-{tag}.npz',trace=tr,final_z=z)
    pv=tr[:,-1,1];ever=tr[:,-1,2]
    stats={'tau_s':8,'omega_input_rad_s':np.pi/16,'temperature_K':300,'nrun':nrun,'dt_s':dt,'nz':nz,'ntheta':ntheta,
      'p_vertical_final':float(pv.mean()),'p_vertical_ever':float(ever.mean()),
      'p_any_final_defect':float((pv>0).mean()),'p_any_ever_defect':float((ever>0).mean()),
      'standard_error_final':float(pv.std(ddof=1)/np.sqrt(nrun)),
      'zero_final_event_95_upper_probability':float(1-.05**(1/nrun)) if not np.any(pv) else None,
      'max_abs_quadrature_seen_um':float(np.max(abs(tr[:,:,3:]))),'grid_exceeded_steps':int(outside.sum()),
      'elapsed_s':time.monotonic()-start,'status':'nonlinear two-quadrature Galerkin closure; prescribed coherent background; no fitted escape cutoff'}
    (OUT/f'summary-{tag}.json').write_text(json.dumps(stats,indent=2))
    mean=pd.DataFrame(tr.mean(axis=0),columns=['t_s','p_vertical','p_vertical_ever','mean_z0','mean_z1'])
    mean['var_z0']=tr[:,:,3].var(axis=0,ddof=1);mean['var_z1']=tr[:,:,4].var(axis=0,ddof=1)
    mean.to_csv(OUT/f'trace-{tag}.csv',index=False);print(stats,flush=True)

@njit(cache=True)
def control_flow(tstart,tau,dt,initial,table,q,E,is_h,group_size,zmin,dz):
    ns=int(np.ceil((tau-tstart)/dt));step=(tau-tstart)/ns;z=initial.copy();ever=np.zeros(len(is_h),np.bool_)
    for k in range(ns):
        u=(tstart+(k+.5)*step)/tau;iq=min(len(q)-2,int(u*(len(q)-1)));fq=u*(len(q)-1)-iq
        qb=(1-fq)*q[iq]+fq*q[iq+1]
        f=interpolate(table,z[0],z[1],u*(len(table)-1),zmin,dz)
        for j in range(len(is_h)):
            s=(qb[0] if is_h[j] else qb[1])+E[j,0]*z[0]+E[j,1]*z[1]
            tf=8*p.HEIGHT*s/100 if abs(s)<5 else -p.STIFFNESS*(s-5*np.sign(s))
            f+=group_size*E[j]*tf/h.GAMMA
        z+=step*f
        for j in range(len(is_h)):
            if not is_h[j]:ever[j]|=qb[1]+E[j]@z<0
    final=q[-1,1]+E[~is_h]@z
    return int((final<0).sum()),int(ever[~is_h].sum()),z

def controls(nz,ntheta):
    test=ModeTest();full,E=basis(test);data=np.load(OUT/f'drift-nz{nz}-ntheta{ntheta}.npz')
    assert np.allclose(E,data['E']);q=test.sol.sol(np.linspace(0,8,7201)).T;zs=data['zgrid'];rows=[]
    for phase,amp in [(0,.942*1.05),(np.pi/10,.8122*1.05),(0,1.6)]:
        d=test.direction(phase);delta=amp*d;z=test.reduced.group_size*E.T@delta
        assert np.max(abs(E@z-delta))<1e-12
        qb=test.sol.sol(28.9*8/90);initial=np.where(test.reduced.is_h,*qb)+delta
        free=continuation(initial,28.9*8/90,8,.00125,*test.reduced.args())
        nv,ever,finalz=control_flow(28.9*8/90,8,.000125,z,data['table'],q,E,test.reduced.is_h,5,zs[0],zs[1]-zs[0])
        # At the very same initial condition, quantify generation of omitted
        # directions. Separate common H/V drift from other nonuniform modes.
        f=n.magnetic(initial,np.sin(np.deg2rad(28.9))**2,*test.reduced.args()[:-2])+c.trap_force(initial)
        fq=np.where(test.reduced.is_h,test.model.force(qb,np.deg2rad(28.9))[0],test.model.force(qb,np.deg2rad(28.9))[1])
        change=(f-fq)/h.GAMMA;proj=5*E.T@change;remain=change-E@proj
        coherent=np.where(test.reduced.is_h,remain[test.reduced.is_h].mean(),remain[~test.reduced.is_h].mean())
        base=np.where(test.reduced.is_h,*qb);eps=1e-5
        def force(ss):return n.magnetic(ss,np.sin(np.deg2rad(28.9))**2,*test.reduced.args()[:-2])+c.trap_force(ss)
        lin=(force(base+eps*delta)-force(base-eps*delta))/(2*eps*h.GAMMA)
        lin_other=lin-E@(5*E.T@lin)
        nonlinear=change-lin
        rows.append({'phase_rad':phase,'initial_max_vertical_amplitude_um':amp,'initial_quadrature_norm_um':float(np.linalg.norm(z)),
          'unrestricted_final_vertical_defects':int(free[0])*5,'unrestricted_peak_vertical_defects':int(free[3])*5,
          'two_quadrature_final_vertical_defects':nv*5,'two_quadrature_ever_vertical_defects':ever*5,
          'two_quadrature_final_norm_um':float(np.linalg.norm(finalz)),
          'initial_projected_drift_norm_um_per_s':float(np.linalg.norm(proj)),
          'initial_generated_coherent_drift_norm_um_per_s':float(np.sqrt(5*np.sum(coherent**2))),
          'initial_generated_other_modes_drift_norm_um_per_s':float(np.sqrt(5*np.sum((remain-coherent)**2))),
          'initial_linear_shape_change_norm_um_per_s':float(np.sqrt(5*np.sum(lin_other**2))),
          'initial_nonlinear_drift_correction_norm_um_per_s':float(np.sqrt(5*np.sum(nonlinear**2)))})
    pd.DataFrame(rows).to_csv(OUT/f'deterministic_closure_checks-nz{nz}-ntheta{ntheta}.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','controls']);parser.add_argument('--nz',type=int,default=33)
    parser.add_argument('--ntheta',type=int,default=181);parser.add_argument('--dt',type=float,default=.00025)
    parser.add_argument('--nrun',type=int,default=2048);parser.add_argument('--seed',type=int,default=81010)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);set_num_threads(4)
    if args.mode=='prepare':prepare(args.nz,args.ntheta)
    elif args.mode=='run':run(args.nz,args.ntheta,args.dt,args.nrun,args.seed)
    else:controls(args.nz,args.ntheta)
