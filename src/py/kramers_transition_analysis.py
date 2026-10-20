"""Cheap instantaneous escape rates on the audited moving tagged potential.

Two-state forward/backward evolution; no new MD and no rate fits. Both the
harmonic Kramers prefactor and an integral high-barrier prefactor are retained.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.optimize import brentq
import coupled_relaxation_analysis as c
OUT=c.p.REPO/'data/figures/kramers-rates'


def landscape(s,u,temp):
    mins=find_peaks(-u)[0];maxs=find_peaks(u)[0]
    plus=mins[s[mins]>0];minus=mins[s[mins]<0]
    blank={'status':'missing_well_or_saddle','positive_minima':len(plus),'negative_minima':len(minus),
           'k_forward_harmonic':np.nan,'k_backward_harmonic':np.nan,
           'k_forward_integral':np.nan,'k_backward_integral':np.nan}
    if not len(plus):return {**blank,'status':'occupied_well_absent'}
    if not len(minus):return {**blank,'status':'opposite_well_absent'}
    a=plus[np.argmin(u[plus])];b=minus[np.argmin(u[minus])]
    peaks=maxs[(maxs>b)&(maxs<a)]
    if not len(peaks):return blank
    z=peaks[np.argmax(u[peaks])];kbT=c.p.KB*temp;D=kbT/c.h.GAMMA
    dx=s[1]-s[0]
    def refine(i):
        curvature=(u[i-1]-2*u[i]+u[i+1])/dx**2
        offset=(u[i-1]-u[i+1])/(2*(u[i-1]-2*u[i]+u[i+1]))*dx
        # Local parabolic refinement; the integral prefactor avoids relying
        # on a harmonic well neighborhood where the trap branches join.
        value=u[i]-.5*curvature*offset**2
        return s[i]+offset,value,curvature
    sa,ua,ca=refine(a);sb,ub,cb=refine(b);sz,uz,cz=refine(z)
    if min(ca,cb)<=0 or cz>=0:return blank
    ha=max(0,uz-ua);hb=max(0,uz-ub)
    barrier_integral=trapezoid(np.exp((u[b:a+1]-u[z])/kbT),s[b:a+1])
    zplus=trapezoid(np.exp(-(u[z:]-u[a])/kbT),s[z:])
    zminus=trapezoid(np.exp(-(u[:z+1]-u[b])/kbT),s[:z+1])
    weight=np.exp(-(u[z:]-u[a])/kbT)/zplus
    mean=trapezoid(s[z:]*weight,s[z:])
    variance=trapezoid((s[z:]-mean)**2*weight,s[z:])
    harmonic=lambda curv,barrier:np.sqrt(curv*abs(cz))/(2*np.pi*c.h.GAMMA)*np.exp(-barrier/kbT)
    integral=lambda zwell,barrier:D/(zwell*barrier_integral)*np.exp(-barrier/kbT)
    # Use grid barriers for quadratures, so their reference offsets cancel.
    kfi=integral(zplus,u[z]-u[a]);kbi=integral(zminus,u[z]-u[b])
    return {'status':'two_wells','positive_minima':len(plus),'negative_minima':len(minus),
            's_occupied_um':sa,'s_opposite_um':sb,'s_saddle_um':sz,
            'barrier_forward_kBT':ha/kbT,'barrier_backward_kBT':hb/kbT,
            'curvature_occupied_pN_per_um':ca,'curvature_opposite_pN_per_um':cb,
            'curvature_saddle_pN_per_um':cz,'intrawell_relaxation_s':c.h.GAMMA/ca,
            'equilibrium_well_variance_um2':variance,
            'variance_based_relaxation_scale_s':variance/D,
            'occupied_cusp_distance_over_thermal_width':abs(abs(sa)-5)/np.sqrt(kbT/ca),
            'k_forward_harmonic':harmonic(ca,ha),'k_backward_harmonic':harmonic(cb,hb),
            'k_forward_integral':kfi,'k_backward_integral':kbi}


def two_state(rates,times):
    p=0.;trace=[0.]
    for j,dt in enumerate(np.diff(times)):
        kf=(rates[j,0]+rates[j+1,0])/2;kb=(rates[j,1]+rates[j+1,1])/2
        total=kf+kb
        if not np.isfinite(total):return np.full(len(times),np.nan)
        if total>0:p=p*np.exp(-total*dt)+(kf/total)*(-np.expm1(-total*dt))
        trace.append(p)
    return np.array(trace)


def evaluate(model,tau,frozen=False,temperature=300,dx=.005,nt=401,write=True):
    label='fixed verticals' if frozen else 'coupled relaxation'
    stem=f'{"fixed" if frozen else "coupled"}-T{temperature:g}-tau{tau:.9g}-dx{dx:g}-nt{nt}'
    target=OUT/f'{stem}.csv'
    df=pd.read_csv(target) if target.exists() else None
    if df is None or 'positive_minima' not in df:
        sol=model.solve(tau,frozen)
        if np.min(sol.y[1])<=0:raise ValueError('Vertical background switches; two-state reference invalid')
        pot=c.PathPotential(model,sol,tau,dx,nt)
        times=np.linspace(0,tau-.05,nt)
        records=[]
        for t in times:
            records.append({'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'model':label,
                            'temperature_K':temperature,'t_s':t,'theta_deg':90*t/tau,
                            'background_s_v_um':float(sol.sol(t)[1]),
                            **landscape(pot.s,pot.energy_at(t),temperature)})
        df=pd.DataFrame(records)
        if write:df.to_csv(target,index=False)
    valid=df.status.eq('two_wells').to_numpy()
    row={'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'model':label,'temperature_K':temperature,
         'dx_um':dx,'nt':nt,'invalid_landscape_frames':int((~valid).sum())}
    indices=np.flatnonzero(valid)
    start=int(indices[0]) if len(indices) else len(valid)
    window_ok=len(indices)>0 and valid[start:].all() and df.status.iloc[:start].eq('opposite_well_absent').all()
    row['rate_evolution_status']='zero occupancy until opposite well appears; two-state thereafter' if window_ok else 'cannot apply two-state rates across missing landscapes'
    row['two_well_start_angle_deg']=float(df.theta_deg.iloc[start]) if len(indices) else np.nan
    for method in ['harmonic','integral']:
        forward=df[f'k_forward_{method}'].to_numpy();backward=df[f'k_backward_{method}'].to_numpy()
        probability=np.full(len(df),np.nan)
        if window_ok:
            probability[:start]=0
            probability[start:]=two_state(np.column_stack([forward[start:],backward[start:]]),df.t_s.to_numpy()[start:])
        df[f'p_{method}']=probability
        row[f'p_final_{method}']=float(probability[-1])
        row[f'forward_hazard_{method}']=float(trapezoid(forward[start:],df.t_s.iloc[start:])) if window_ok else np.nan
    if window_ok:
        peak=int(np.nanargmax(df.k_forward_integral))
        lograte=np.log(np.maximum(df.k_forward_integral.iloc[start:],1e-300))
        ratio=np.full(len(df),np.nan)
        ratio[start:]=df.variance_based_relaxation_scale_s.iloc[start:]*np.abs(np.gradient(lograte,df.t_s.iloc[start:]))
        active=df.k_forward_integral>.1*df.k_forward_integral.max()
        row.update({'minimum_forward_barrier_kBT':float(df.barrier_forward_kBT.min()),
                    'angle_of_peak_forward_rate_deg':float(df.theta_deg.iloc[peak]),
                    'barrier_at_peak_rate_kBT':float(df.barrier_forward_kBT.iloc[peak]),
                    'drive_to_relaxation_ratio_max_active':float(ratio[active].max()),
                    'background_distance_from_well_min_over_sigma_at_peak':float(abs(df.background_s_v_um.iloc[peak]-df.s_occupied_um.iloc[peak])/np.sqrt(df.equilibrium_well_variance_um2.iloc[peak])),
                    'cusp_distance_over_width_at_peak':float(df.occupied_cusp_distance_over_thermal_width.iloc[peak])})
        df['drive_to_relaxation_ratio']=ratio
    if write:df.to_csv(target,index=False)
    print(row,flush=True)
    return row,df


def verify():
    # Symmetric smooth quartic with increasing barrier: the quadrature and
    # harmonic prefactors should approach each other, independent of our lattice.
    s=np.linspace(-2,2,16001);ratios=[]
    for height in [10,40,100]:
        u=height*c.p.KB*300*(1-s*s)**2
        r=landscape(s,u,300)
        ratios.append(r['k_forward_integral']/r['k_forward_harmonic'])
    assert abs(ratios[-1]-1)<.02 and abs(ratios[-1]-1)<abs(ratios[0]-1)
    t=np.linspace(0,2,101);pr=two_state(np.tile([.3,.1],(101,1)),t)
    assert np.isclose(pr[-1],.75*(1-np.exp(-.8)))
    return {'quartic_integral_to_harmonic_rate_ratios':ratios,'constant_two_state_solution':True}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--taus',type=float,nargs='+',default=[8,15,30,64.7752533928]);parser.add_argument('--nt',type=int,default=401);parser.add_argument('--dx',type=float,default=.005)
    parser.add_argument('--temperature',type=float,default=300)
    parser.add_argument('--coupled-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    print('Independent checks:',verify(),flush=True)
    model=c.CoupledModel();rows=[]
    for tau in args.taus:
        for frozen in ([False] if args.coupled_only else [True,False]):
            row,df=evaluate(model,tau,frozen,args.temperature,args.dx,args.nt);rows.append(row)
    target=OUT/f'summary-T{args.temperature:g}-dx{args.dx:g}-nt{args.nt}.csv'
    result=pd.DataFrame(rows)
    if target.exists():result=pd.concat([pd.read_csv(target),result]).drop_duplicates(['tau_s','model','temperature_K'],keep='last')
    result.to_csv(target,index=False)
