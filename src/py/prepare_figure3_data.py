"""Export compact Fig. 3 curves, preserving the original 300 K reductions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'data/test11/10'
TEMPERATURE_SOURCE=REPO.parent/'latex/summary-answer/data/final_observables.csv'
OUT=REPO/'data/figures/paper';OUT.mkdir(parents=True,exist_ok=True)
# Retain the original rounded-frequency grouping for the upper panel.
raw=pd.read_csv(BASE/'average_counts.csv',index_col=['field','time'])
raw['omega']=np.around(np.pi/2/raw.total_time.to_numpy(),4)
data20=raw.xs(20,level='field').reset_index()
max_times={w:g.time.max() for w,g in data20.groupby('omega')}
counts=pd.concat(data20.query('omega==@w and time==@t') for w,t in max_times.items()).sort_values('omega')
population=pd.DataFrame(dict(temperature_K=300.,field_mT=20.,rotation_duration_s=counts.total_time,
    observation_time_s=counts.time,omega_rad_s=counts.omega,p_abs_q4=counts.I+counts.VI,
    p_abs_q2=counts.II+counts.V,p_q0=counts.III+counts.IV))
assert np.allclose(population[['p_abs_q4','p_abs_q2','p_q0']].sum(axis=1),1)
population.to_csv(OUT/'fig3_vertex_populations.csv',index=False)
# The original notebook takes absolute kappa before averaging, selects the last
# rounded-angle sample for each duration, and normalizes by 4*100 vertices.
kappa=pd.read_csv(BASE/'kappa.csv')
assert kappa.field.eq(20).all()
kappa['theta']=np.round(kappa.t*np.pi/2/kappa.total_time*180/np.pi,4)
kappa['order']=abs(kappa.kappa)/400
warm=kappa.groupby(['total_time','theta'],as_index=False).agg(
    order_mean=('order','mean'),order_std=('order','std'),n_runs=('order','size'),observation_time_s=('t','first'))
warm=warm.loc[warm.groupby('total_time').theta.idxmax()].copy()
assert warm.n_runs.eq(10).all()
warm['temperature_K']=300.;warm['field_mT']=20.;warm['omega_rad_s']=np.pi/2/warm.total_time
warm['order_sem']=warm.order_std/np.sqrt(warm.n_runs)
warm['source']='historical-paper-300K: repo/data/test11/10/kappa.csv'
warm=warm.rename(columns={'total_time':'rotation_duration_s'})
extra=pd.read_csv(TEMPERATURE_SOURCE).query('field==20 and temperature_K in [0.1,1,10,100]').copy()
assert set(extra.temperature_K)=={.1,1.,10.,100.} and extra.n.eq(10).all()
extra=extra.rename(columns={'field':'field_mT','total_time':'rotation_duration_s','omega':'omega_rad_s',
    'kappa':'order_mean','kappa_std':'order_std','kappa_sem':'order_sem','n':'n_runs','campaign':'source'})
extra['observation_time_s']=extra.rotation_duration_s-.05
columns=['temperature_K','field_mT','rotation_duration_s','observation_time_s','omega_rad_s',
         'order_mean','order_std','order_sem','n_runs','source']
order=pd.concat([warm[columns],extra[columns]],ignore_index=True).sort_values(['temperature_K','omega_rad_s'])
assert not order.duplicated(['temperature_K','omega_rad_s']).any()
assert order.order_mean.between(0,1).all()
order.to_csv(OUT/'fig3_charge_order.csv',index=False)
sources=[]
for p in [BASE/'average_counts.csv',BASE/'kappa.csv',TEMPERATURE_SOURCE]:
 sources.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'fig3_data_provenance.json').write_text(json.dumps(dict(
    field_mT=20,vertices=100,particles=200,n_runs=10,
    upper_panel='original 300 K populations; omega rounded to 4 decimals, original endpoint grouping',
    lower_panel='original 300 K curve plus 0.1, 1, 10, 100 K endpoint means; no pooling across campaigns',
    normalization='mean(abs(raw staggered charge sum))/(4*100)',
    uncertainties='order_std is sample SD; order_sem=order_std/sqrt(n_runs); saved but not plotted',
    curves='solid lines connect samples; no interpolation or fit',
    sources=sources),indent=2)+'\n')
print(order.groupby('temperature_K').size().to_string())
print('Saved:',OUT)
