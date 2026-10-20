"""Resolve a near-threshold deterministic escape into orthogonal motions.

Paired unperturbed continuations remove common discretization drift. No new
noise histories, boundary fits, or changes to existing figures.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/frequencygs-matplotlib')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/frequencygs-numba')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
import pandas as pd
from numba import njit
from pathway_selection_theory import ModeTest
from projected_mode_noise import basis
import nonlinear_collective_analysis as n
import collective_horizontal_analysis as h
import rotating_potential_analysis as p
OUT=p.REPO/'data/figures/deterministic-pathway-diagnostic'

@njit(cache=True)
def trajectory(initial,tstart,tau,dt,indptr,source,rx,ry,ex,ey,ax,ay,mult,is_h,group_size):
    ns=int(np.ceil((tau-tstart)/dt));step=(tau-tstart)/ns
    stride=max(1,int(round(.005/step)));nf=ns//stride+1+(ns%stride!=0)
    times=np.zeros(nf);states=np.zeros((nf,len(initial)));s=initial.copy()
    times[0]=tstart;states[0]=s;frame=1;first=np.nan;first_state=s.copy()
    for k in range(ns):
        t=tstart+(k+.5)*step
        f=n.magnetic(s,np.sin(np.pi*t/(2*tau))**2,indptr,source,rx,ry,ex,ey,ax,ay,mult)
        for j in range(len(s)):s[j]=n.trap_resolvent(s[j]+step*f[j]/h.GAMMA,step)
        if np.isnan(first) and np.any(s[~is_h]<0):first=tstart+(k+1)*step;first_state=s.copy()
        if (k+1)%stride==0 or k==ns-1:
            times[frame]=tstart+(k+1)*step;states[frame]=s;frame+=1
    return times,states,first,first_state

def decompose(test,E,s,background):
    d=s-background;mh=d[:,test.reduced.is_h].mean(axis=1);mv=d[:,~test.reduced.is_h].mean(axis=1)
    coherent=np.where(test.reduced.is_h[None,:],mh[:,None],mv[:,None])
    z=5*d@E;mode=z@E.T;other=d-coherent-mode
    total=5*np.sum(d*d,axis=1);nc=5*np.sum(coherent**2,axis=1);nm=np.sum(z*z,axis=1);nr=5*np.sum(other**2,axis=1)
    assert np.max(abs(total-nc-nm-nr))<1e-9
    # Full unitary Fourier transform: retain the entire selected conjugate
    # sector, not just its initial two-dimensional internal cell pattern.
    full=d[:,test.reduced.mapping]
    arranged=full[:,test.lattice.order].reshape(len(d),5,5,8)
    ft=np.fft.fftn(arranged,axes=(1,2),norm='ortho')
    kx,ky=test.lattice.wave[test.kk];pair=np.sum(abs(ft[:,kx,ky,:])**2,axis=1)+np.sum(abs(ft[:,(-kx)%5,(-ky)%5,:])**2,axis=1)
    zero=np.sum(abs(ft[:,0,0,:])**2,axis=1)
    new=np.maximum(0,total-pair-zero);within=np.maximum(0,pair-nm);zero_other=np.maximum(0,zero-nc)
    assert np.max(abs(nr-within-zero_other-new))<1e-9
    mask=~test.reduced.is_h;recon4=background+coherent+mode;recon2=background+mode
    return pd.DataFrame({'mean_h_shift_um':mh,'mean_v_shift_um':mv,'total_norm_sq_um2':total,
      'coherent_norm_sq_um2':nc,'original_mode_norm_sq_um2':nm,'other_pattern_norm_sq_um2':nr,
      'within_original_wavevector_shape_norm_sq_um2':within,'zero_wavevector_other_shape_norm_sq_um2':zero_other,
      'new_wavevector_norm_sq_um2':new,'coherent_rms_um':np.sqrt(nc/200),'original_mode_rms_um':np.sqrt(nm/200),
      'other_pattern_rms_um':np.sqrt(nr/200),'minimum_vertical_actual_um':s[:,mask].min(axis=1),
      'minimum_vertical_mean_plus_mode_um':recon4[:,mask].min(axis=1),'minimum_vertical_mode_only_um':recon2[:,mask].min(axis=1),
      'actual_vertical_defects':5*np.sum(s[:,mask]<0,axis=1),'mean_plus_mode_vertical_defects':5*np.sum(recon4[:,mask]<0,axis=1)})

def main():
    OUT.mkdir(parents=True,exist_ok=True);test=ModeTest();_,E=basis(test)
    start=28.9*8/90;initial_bg=np.where(test.reduced.is_h,*test.sol.sol(start));rows=[];events=[];checks=[]
    brackets=pd.read_csv(p.REPO/'data/figures/pathway-selection-theory/deterministic_brackets-theta28.9-dt0.00125.csv')
    for dt in [.00125,.000625]:
        times,bg,_,_=trajectory(initial_bg,start,8,dt,*test.reduced.args())
        reference=np.where(test.reduced.is_h[None,:],test.sol.sol(times)[0,:,None],test.sol.sol(times)[1,:,None])
        checks.append({'dt_s':dt,'max_unperturbed_deviation_from_Radau_um':float(abs(bg-reference).max()),
                       'unperturbed_vertical_defects':int(5*np.sum(bg[-1,~test.reduced.is_h]<0))})
        assert np.max(np.ptp(bg[:,test.reduced.is_h],axis=1))<1e-10
        assert np.max(np.ptp(bg[:,~test.reduced.is_h],axis=1))<1e-10
        for _,bracket in brackets.iterrows():
            phase=bracket.phase_rad;threshold=(bracket.amplitude_lower_no_switch_um+bracket.amplitude_upper_switch_um)/2
            for side in ['below','above']:
                amp=threshold+(.005 if side=='above' else -.005)
                tt,ss,first,first_state=trajectory(initial_bg+amp*test.direction(phase),start,8,dt,*test.reduced.args())
                assert np.array_equal(tt,times)
                f=decompose(test,E,ss,bg);f.insert(0,'t_s',tt);f.insert(1,'theta_deg',tt*90/8)
                tag=f'phase{phase:.6f}-{side}-dt{dt:g}'
                f.to_csv(OUT/f'trace-{tag}.csv',index=False)
                np.savez_compressed(OUT/f'path-{tag}.npz',times=tt,states=ss,unperturbed=bg)
                row={'phase_rad':phase,'seed_amplitude_um':amp,'threshold_side':side,'dt_s':dt,
                  'first_vertical_angle_deg':float(first*90/8),'final_vertical_defects':int(f.actual_vertical_defects.iloc[-1]),
                  'final_mean_plus_mode_defects':int(f.mean_plus_mode_vertical_defects.iloc[-1]),'source':f'trace-{tag}.csv'}
                if np.isfinite(first):
                    # Evaluate components at the actual crossing step, not at
                    # the coarser output time. Background remains spatially
                    # coherent; linearly interpolate its short stored interval.
                    bf=np.array([np.interp(first,tt,bg[:,j]) for j in range(40)])
                    event=decompose(test,E,first_state[None,:],bf[None,:]).iloc[0].to_dict()
                    event.update(row);events.append(event)
                    for key in ['coherent','original_mode','other_pattern']:
                        row[f'{key}_fraction_at_first_switch']=event[f'{key}_norm_sq_um2']/event['total_norm_sq_um2']
                    row['mean_h_shift_at_first_um']=event['mean_h_shift_um'];row['mean_v_shift_at_first_um']=event['mean_v_shift_um']
                    row['minimum_vertical_mean_plus_mode_at_first_um']=event['minimum_vertical_mean_plus_mode_um']
                    row['new_wavevector_fraction_at_first']=event['new_wavevector_norm_sq_um2']/event['total_norm_sq_um2']
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(events).to_csv(OUT/'first_switch_decomposition.csv',index=False)
    (OUT/'checks.json').write_text(json.dumps({'paired_unperturbed_checks':checks,'orthogonal_partition_tolerance_um2':1e-9,
      'seed_angle_deg':28.9,'tau_input_s':8,'noise':False,'normalization':'squared Euclidean displacement over 200 sites; not an energy'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
