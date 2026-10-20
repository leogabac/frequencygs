"""Store the first theoretical boundary separately from empirical estimates."""
import json,hashlib
from pathlib import Path
import vertical_competition_analysis as v
out=v.OUT
root=json.loads((out/'warm_boundary_root.json').read_text())
evals=v.pd.read_csv(out/'warm_boundary_root_evaluations.csv')
lo=float(evals.loc[evals.p_vertical_final>=.01,'omega_rad_s'].max())
hi=float(evals.loc[evals.p_vertical_final<.01,'omega_rad_s'].min())
record={'temperature_K':300,'field_mT':20,
        'model':'coherent horizontal ODE; frozen other verticals; tagged Smoluchowski',
        'omega_estimate_rad_s':root['omega_rad_s'],
        'omega_root_bracket_lower_rad_s':lo,'omega_root_bracket_upper_rad_s':hi,
        'omega_error_minus_rad_s':max(0,root['omega_rad_s']-lo),
        'omega_error_plus_rad_s':max(0,hi-root['omega_rad_s']),
        'error_type':'root-finding bracket only; excludes mesh error and model bias',
        'status':'fails empirical warm bracket','dx_um':root['dx_um'],'dt_s':root['dt_s'],
        'source':'../figures/vertical-competition/warm_boundary_root_evaluations.csv'}
v.pd.DataFrame([record]).to_csv(v.p.REPO/'data/analysis/theoretical_boundary_estimates.csv',index=False)
metadata={'result':'warm boundary disagrees with data; cold PDE unconverged',
          'model_sources_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
             [Path(v.__file__),Path(v.h.__file__),Path(v.p.__file__),v.p.STATE,v.p.REPO/'src/parameters.py']},
          'warm_parameters':root,'parameter_quantities_verified':v.p.verify_parameter_quantities(),
          'solver_checks':v.verification()}
(out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Stored theoretical warm estimate separately:',record['omega_estimate_rad_s'])
