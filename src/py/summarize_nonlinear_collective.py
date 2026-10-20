"""Run-level uncertainty and sampled transition brackets; keep empirical data."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import nonlinear_collective_analysis as a

def main():
    rng=np.random.default_rng(20261010);rows=[];samples={};runfractions={};preserved={}
    prior=json.loads((a.p.REPO/'data/figures/kramers-rates/provenance.json').read_text())
    for path,digest in prior['preserved_artifacts_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        preserved[path]=digest
    for f in sorted(a.OUT.glob('summary-*.json')):
        row=json.loads(f.read_text());row.pop('checks',None)
        data=np.load(a.OUT/row['source']);final=data['final'];ref=data['ref'];is_h=pd.read_csv(a.p.STATE).dx.abs().to_numpy()[ref]>.5
        per=(final[:,~is_h]<0).mean(axis=1);n=len(per)
        draws=per[rng.integers(n,size=(5000,n))].mean(axis=1)
        if np.all(per==0):
            low=0.;high=1-.05**(1/n)
            interval='zero-event conservative one-sided 95% run-level bound; site dependence allowed'
        else:
            low,high=np.quantile(draws,[.025,.975]);interval='95% bootstrap over independent model runs; site dependence retained'
        row.update({'p_ci_lower':low,'p_ci_upper':high,'p_error_minus':per.mean()-low,'p_error_plus':high-per.mean(),'interval_type':interval,'runs_with_vertical_switches':int((per>0).sum())})
        rows.append(row);samples[row['source']]=draws;runfractions[row['source']]=per
    frame=pd.DataFrame(rows);frame.to_csv(a.OUT/'summary_with_intervals.csv',index=False)
    main=frame.query('nsite==200 and temperature_K>0').sort_values(['nrun','dt_requested_s'],ascending=[False,True]).drop_duplicates(['temperature_K','tau_s'])
    main.to_csv(a.OUT/'selected_results.csv',index=False)
    estimates=[]
    for temp,g in main.groupby('temperature_K'):
        g=g.sort_values('omega_rad_s');below=g[g.p_vertical_final<.01];above=g[g.p_vertical_final>=.01]
        if below.empty or above.empty:continue
        lo=above.iloc[-1];hi=below.iloc[0]
        if lo.omega_rad_s>=hi.omega_rad_s:continue
        def interpolate(pl,ph):
            return lo.omega_rad_s+(.01-pl)/(ph-pl)*(hi.omega_rad_s-lo.omega_rad_s)
        center=interpolate(lo.p_vertical_final,hi.p_vertical_final)
        bl=samples[lo.source];bh=samples[hi.source]
        # Identical seeds across protocols are common random numbers. Resample
        # matching run indices together, including unmatched extra runs.
        if lo.seed==hi.seed:
            pl=runfractions[lo.source];ph=runfractions[hi.source];shared=min(len(pl),len(ph))
            indices=rng.integers(shared,size=(5000,shared))
            bl=pl[indices].sum(axis=1);bh=ph[indices].sum(axis=1)
            for per,draw in [(pl,bl),(ph,bh)]:
                extra=len(per)-shared
                if extra:draw+=per[rng.integers(shared,len(per),size=(5000,extra))].sum(axis=1)
            bl/=len(pl);bh/=len(ph)
        valid=(bl>=.01)&(bh<.01)&(bl>bh)
        roots=interpolate(bl[valid],bh[valid]);ci=np.quantile(roots,[.025,.975]) if len(roots) and lo.p_vertical_final>0 and hi.p_vertical_final>0 else [np.nan,np.nan]
        estimates.append({'temperature_K':temp,'field_mT':20,'model':'200-coordinate nonlinear along-trap Langevin; implicit trap branches',
          'omega_estimate_rad_s':center,'omega_root_bracket_lower_rad_s':lo.omega_rad_s,'omega_root_bracket_upper_rad_s':hi.omega_rad_s,
          'omega_error_minus_rad_s':center-lo.omega_rad_s,'omega_error_plus_rad_s':hi.omega_rad_s-center,
          'error_type':'sampled frequency bracket; excludes discretization and model bias',
          'omega_bootstrap_lower_rad_s':ci[0],'omega_bootstrap_upper_rad_s':ci[1],'bootstrap_fraction_retaining_bracket':valid.mean(),'bootstrap_type':'paired run indices for shared seeds; conditional on sampled bracket; unavailable at zero-event endpoint',
          'status':'nonlinear reduced-coordinate estimate; Monte Carlo and step checks reported; no full-MD validation',
          'dt_s':max(lo.dt_requested_s,hi.dt_requested_s),'source':'../figures/nonlinear-collective/selected_results.csv'})
    est=pd.DataFrame(estimates);est.to_csv(a.OUT/'boundary_estimates.csv',index=False)
    if len(est):
        target=a.p.REPO/'data/analysis/theoretical_boundary_estimates.csv';old=pd.read_csv(target)
        pd.concat([old,est],ignore_index=True).drop_duplicates(['temperature_K','field_mT','model'],keep='last').to_csv(target,index=False)
    checks=frame.query('nsite==200 and tau_s==8 and temperature_K==300').sort_values('dt_requested_s')
    checks.to_csv(a.OUT/'warm_step_comparison.csv',index=False)
    refinement=[]
    for temp,tau in [(300,8),(.1,40)]:
        g=frame.query('nsite==200 and temperature_K==@temp and tau_s==@tau').sort_values('dt_requested_s',ascending=False)
        for i in range(len(g)-1):
            coarse=g.iloc[i];fine=g.iloc[i+1];pc=runfractions[coarse.source];pf=runfractions[fine.source]
            if coarse.seed==fine.seed and len(pc)==len(pf):
                se=(pf-pc).std(ddof=1)/np.sqrt(len(pc));method='paired run difference for shared seeds'
            else:
                se=np.sqrt(pc.var(ddof=1)/len(pc)+pf.var(ddof=1)/len(pf));method='independent run estimates'
            refinement.append({'temperature_K':temp,'tau_s':tau,'coarse_dt_s':coarse.dt_requested_s,'fine_dt_s':fine.dt_requested_s,
                 'fine_minus_coarse_probability':pf.mean()-pc.mean(),'difference_standard_error':se,'sampling_comparison_method':method})
    pd.DataFrame(refinement).to_csv(a.OUT/'stochastic_step_checks.csv',index=False)
    sources=[a.p.STATE,a.p.REPO/'src/py/nonlinear_collective_analysis.py',a.p.REPO/'src/py/summarize_nonlinear_collective.py']
    manifest={'model':'overdamped along-trap coordinates; nonlinear magnetic pair forces; implicit exact scalar trap resolvent',
      'bootstrap_draws':5000,'bootstrap_rng_seed':20261010,'selected_result_rule':'largest ensemble per protocol, then finest step if ensemble size ties',
      'noise':'D=kB*T/gamma, gamma fixed; symmetry-restricted group coordinate has D/group_size',
      'geometry':'initial ordered size10 state; 200 um dipolar cutoff; all planar periodic images',
      'initial_state':'s_i=5 um for every run; no thermal pre-equilibration',
      'observable':'s_vertical<0 at tau-.05 s; first-ever crossings also saved; matches ideal-state displacement criterion',
      'method':'explicit magnetic Euler drift and Brownian increment followed by implicit trap drift',
      'limitations':['transverse and out-of-plane coordinates omitted','frequency interpolation is approximate','time-step comparisons use independently sampled paths, despite equal RNG seed','bootstrap covers sampling, not force-model or discretization bias'],
      'preserved_artifacts_sha256':preserved,'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (a.OUT/'provenance.json').write_text(json.dumps(manifest,indent=2))
    print(main[['temperature_K','tau_s','p_vertical_final','p_ci_lower','p_ci_upper','dt_requested_s','nrun']].to_string(index=False));print(est.to_string(index=False))
if __name__=='__main__':main()
