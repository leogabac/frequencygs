"""Keep formal Kramers estimates separate from empirical and validated results."""
import json,hashlib
from pathlib import Path
import kramers_transition_analysis as k
summary=k.pd.read_csv(k.OUT/'summary-T300-dx0.005-nt401.csv')
rows=[]
for method in ['harmonic','integral']:
    data=summary.query('model=="coupled relaxation"').sort_values('omega_rad_s')
    prob=data[f'p_final_{method}'].to_numpy();omega=data.omega_rad_s.to_numpy()
    brackets=k.np.flatnonzero((prob[:-1]>=.01)&(prob[1:]<.01))
    if not len(brackets):continue
    i=int(brackets[0]);lo,hi=omega[i:i+2]
    frac=(k.np.log(.01)-k.np.log(prob[i]))/(k.np.log(prob[i+1])-k.np.log(prob[i]))
    estimate=float(lo+frac*(hi-lo))
    rows.append({'temperature_K':300,'field_mT':20,'model':f'formal Kramers {method} on coupled audited background',
                 'omega_estimate_rad_s':estimate,'omega_root_bracket_lower_rad_s':lo,'omega_root_bracket_upper_rad_s':hi,
                 'omega_error_minus_rad_s':estimate-lo,'omega_error_plus_rad_s':hi-estimate,
                 'error_type':'sampled rate-model bracket; excludes asymptotic error and model bias',
                 'status':'uncontrolled instantaneous-rate approximation; fails empirical warm bracket',
                 'dx_um':.005,'dt_s':k.np.nan,'source':'../figures/kramers-rates/summary-T300-dx0.005-nt401.csv'})
path=k.c.p.REPO/'data/analysis/theoretical_boundary_estimates.csv'
old=k.pd.read_csv(path)
old=old.loc[~old.model.isin([r['model'] for r in rows])]
k.pd.concat([old,k.pd.DataFrame(rows)],ignore_index=True).to_csv(path,index=False)
(k.OUT/'formal_boundary_brackets.json').write_text(json.dumps(rows,indent=2)+'\n')
preserved=json.loads((k.OUT/'preserved_artifacts_sha256.json').read_text())
for name,digest in preserved.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
files=[Path(k.__file__),Path(k.c.__file__),Path(__file__),k.c.p.STATE,
       k.c.p.REPO/'data/analysis/left-boundary-timescales/timescales.csv']
(k.OUT/'provenance.json').write_text(json.dumps({'temperature_K':300,'gamma_pN_s_per_um':k.c.h.GAMMA,
 'method':'instantaneous forward/backward high-barrier rates; exact constant-rate two-state step',
 'integral_prefactor':'D exp(-DeltaU/kBT)/(Z_occupied I_barrier); handles nonharmonic wells in the high-barrier approximation',
 'missing_well_policy':'zero occupancy before opposite minimum forms; do not extrapolate rates across later missing wells',
 'interpretation':'formal diagnostics; neither adiabatic separation nor harmonic well approximation is controlled',
 'rate_model_checks':k.verify(),'formal_boundary_brackets':rows,
 'sources_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
 'preserved_artifacts_sha256':preserved},indent=2)+'\n')
print(k.pd.DataFrame(rows)[['model','omega_estimate_rad_s','omega_root_bracket_lower_rad_s','omega_root_bracket_upper_rad_s']].to_string(index=False))
