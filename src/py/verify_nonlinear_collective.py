"""Deterministic limit and stochastic time-step checks for the reduced model."""
import json
import numpy as np
import pandas as pd
from numba import set_num_threads
import nonlinear_collective_analysis as a

def main():
    a.OUT.mkdir(parents=True,exist_ok=True);model=a.Reduction(1,5);coupled=a.c.CoupledModel();sol=coupled.solve(8)
    rows=[];set_num_threads(2)
    for dt in [.005,.0025,.00125]:
        trace,final=a.ensemble(8,0,dt,1,61010,*model.args())
        f=trace[0];q=sol.sol(f[:,0]);rh=abs(q[0]-5)/10;rv=abs(q[1]-5)/10
        row={'dt_s':dt,'max_Rh_error_vs_Radau':float(np.max(abs(f[:,3]-rh))),
             'max_Rv_error_vs_Radau':float(np.max(abs(f[:,4]-rv))),
             'final_vertical_fraction':float((final[:,~model.is_h]<0).mean())};rows.append(row)
        print(row,flush=True)
    pd.DataFrame(rows).to_csv(a.OUT/'deterministic_step_checks.csv',index=False)
    set_num_threads(1);one=a.ensemble(.2,300,.005,4,61010,*model.args())
    set_num_threads(2);two=a.ensemble(.2,300,.005,4,61010,*model.args())
    assert np.array_equal(one[0],two[0]) and np.array_equal(one[1],two[1])
    assert np.isclose(one[0][0,-1,0],.15) and np.all(np.diff(one[0][0,:,0])>0)
    (a.OUT/'parallel_rng_check.json').write_text(json.dumps({'same_seed_one_vs_two_threads_identical':True,'trace_ends_at_requested_observation_time':True},indent=2))
    assert rows[-1]['max_Rh_error_vs_Radau']<rows[0]['max_Rh_error_vs_Radau']
    assert rows[-1]['max_Rv_error_vs_Radau']<rows[0]['max_Rv_error_vs_Radau']
    assert rows[-1]['max_Rh_error_vs_Radau']<.002 and rows[-1]['max_Rv_error_vs_Radau']<.002
if __name__=='__main__':main()
