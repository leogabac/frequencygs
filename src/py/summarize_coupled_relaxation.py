"""Record the coupled-model left boundary, keeping earlier estimates intact."""
import json,hashlib
from pathlib import Path
import coupled_relaxation_analysis as c
out=c.OUT
root=json.loads((out/'warm_boundary.json').read_text())
evals=c.pd.read_csv(out/'boundary_evaluations.csv').query('dx_um==.005 and dt_s==.005 and nt==901')
lo=float(evals.loc[evals.p_vertical_final>=.01,'omega_rad_s'].max())
hi=float(evals.loc[evals.p_vertical_final<.01,'omega_rad_s'].min())
model='coupled deterministic H/V relaxation; tagged Smoluchowski; audited finite-cutoff forces'
record={'temperature_K':300,'field_mT':20,'model':model,'omega_estimate_rad_s':root['omega_rad_s'],
'omega_root_bracket_lower_rad_s':lo,'omega_root_bracket_upper_rad_s':hi,
'omega_error_minus_rad_s':max(0,root['omega_rad_s']-lo),'omega_error_plus_rad_s':max(0,hi-root['omega_rad_s']),
'error_type':'root-finding bracket only; excludes mesh error and model bias','status':'fails empirical warm left-boundary bracket',
'dx_um':.005,'dt_s':.005,'source':'../figures/coupled-relaxation/boundary_evaluations.csv'}
path=c.p.REPO/'data/analysis/theoretical_boundary_estimates.csv'
old=c.pd.read_csv(path)
old=old.loc[old.model!=model]
c.pd.concat([old,c.pd.DataFrame([record])],ignore_index=True).to_csv(path,index=False)
preserved=json.loads((out/'preserved_artifacts_sha256.json').read_text())
for name,digest in preserved.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
files=[Path(c.__file__),c.p.STATE,c.p.REPO/'src/parameters.py',Path(__file__),
       c.p.REPO/'data/analysis/left-boundary-timescales/averaged_motion.csv',
       c.p.REPO/'data/analysis/left-boundary-timescales/timescales.csv']
metadata={'model':model,'temperature_K':300,'gamma_pN_s_per_um':c.h.GAMMA,
'coordinates':'s_h,s_v: positive into initially occupied well; all 100 particles in each orientation share its signed coordinate',
'noise':'none in coherent background; thermal diffusion only for the tagged vertical particle',
'equation':'gamma ds_a/dt = projected total force on orientation a / 100',
'interaction_convention':'all xy periodic images inside rc=200 um; zero LJ; planar forces; no z image layers',
'force_coefficient_pN_um4':c.COEFFICIENT_MD,'field_conversion_factor':c.COEFFICIENT_MD/c.p.COEFFICIENT,
'force_audit':'bundled magcolloids LAMMPS; 4 configurations; two run 0 calls each, no MD timesteps',
'probability':'instantaneous tagged vertical mass at s<0, at tau-0.05 s; includes returns',
'root_result':root,'parameter_quantities_verified':c.p.verify_parameter_quantities(),
'sources_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
'prior_artifacts_preserved':True,'preserved_artifacts_sha256':preserved}
(out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Appended coupled estimate; all previous figures and empirical estimates remain unchanged.')
