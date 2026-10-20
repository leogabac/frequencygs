"""Save sampled Gaussian diagnostic brackets separately from measured boundaries."""
import json,hashlib
import numpy as np
import pandas as pd
import collective_stability_analysis as a

def main():
    fine=pd.read_csv(a.OUT/'summary-n3600.csv');coarse=pd.read_csv(a.OUT/'summary-n1800.csv')
    joined=fine.merge(coarse,on='tau_s',suffixes=('_fine','_coarse'))
    check=joined[['tau_s']].copy()
    for col in ['peak_gaussian_tail_300K','peak_gaussian_tail_0p1K','sigma_v_at_peak_um','unstable_window_max_singular_gain']:
        check[col+'_relative_change']=abs(joined[col+'_fine']-joined[col+'_coarse'])/np.maximum(abs(joined[col+'_fine']),1e-300)
    check.to_csv(a.OUT/'step_refinement.csv',index=False)
    rows=[]
    for temp,col in [(300,'peak_gaussian_tail_300K'),(.1,'peak_gaussian_tail_0p1K')]:
        f=fine.sort_values('omega_rad_s')
        lo=f.loc[f[col]>=.01].iloc[-1];hi=f.loc[f[col]<.01].iloc[0]
        center=np.interp(np.log(.01),np.log([hi[col],lo[col]]),[hi.omega_rad_s,lo.omega_rad_s])
        rows.append({'temperature_K':temp,'field_mT':20,'model':'200 along-trap linear Langevin modes; peak instantaneous Gaussian tail',
                     'omega_estimate_rad_s':center,'omega_root_bracket_lower_rad_s':lo.omega_rad_s,'omega_root_bracket_upper_rad_s':hi.omega_rad_s,
                     'omega_error_minus_rad_s':center-lo.omega_rad_s,'omega_error_plus_rad_s':hi.omega_rad_s-center,
                     'error_type':'sampled diagnostic bracket; not confidence interval; nonlinear model bias excluded',
                     'status':'uncontrolled Gaussian diagnostic; outer trap branch reached; not a final-crossing prediction',
                     'source':'../figures/collective-stability/summary-n3600.csv'})
    estimates=pd.DataFrame(rows);estimates.to_csv(a.OUT/'gaussian_diagnostic_brackets.csv',index=False)
    target=a.p.REPO/'data/analysis/theoretical_boundary_estimates.csv'
    old=pd.read_csv(target)
    pd.concat([old,estimates],ignore_index=True).drop_duplicates(['temperature_K','field_mT','model'],keep='last').to_csv(target,index=False)
    sources=[a.p.STATE,a.p.REPO/'data/analysis/omega_vs_temperature.csv',a.p.REPO/'data/analysis/left-boundary-timescales/averaged_motion.csv',
             a.p.REPO/'src/py/collective_stability_analysis.py']
    manifest={'description':'local force Jacobian, exact frozen-matrix Gaussian noise steps; no MD, no fitted constants',
              'temperature_convention':'noise temperature only; drag fixed; C(T)=T/300*C(300K)',
              'coordinates':'one signed coordinate per trap; transverse and out-of-plane fluctuations omitted',
              'geometry':'200 sites; 25 cells of 8 sites; all planar periodic images inside 200 um',
              'initial_covariance':'zero, common deterministic initial state',
              'force_derivative':'inside active interaction cutoff and local trap branch; no cutoff impulses',
              'step_method':'midpoint frozen Hermitian Jacobian; matrix exponential and exact constant-J noise integral; split at background trap joins',
              'gain_window':'first contiguous interval with largest eigenvalue > 0; maximum singular value of time-ordered propagator',
              'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (a.OUT/'provenance.json').write_text(json.dumps(manifest,indent=2))
    print(estimates.to_string(index=False));print(check.to_string(index=False))
if __name__=='__main__':main()
