"""Find the warm 1% crossing for the corrected coupled background."""
import json
from scipy.optimize import brentq
import coupled_relaxation_analysis as c

def main():
    model=c.CoupledModel()
    previous=c.OUT/'boundary_evaluations.csv'
    records=c.pd.read_csv(previous).to_dict('records') if previous.exists() else []
    def evaluate(tau,dx=.005,dt=.005,nt=901):
        for row in records:
            if abs(row['tau_s']-tau)<1e-9 and row['dx_um']==dx and row['dt_s']==dt and row['nt']==nt:
                print('Reused evaluation:',tau,row['p_vertical_final'],flush=True)
                return row['p_vertical_final']
        sol=model.solve(tau)
        assert c.np.min(sol.y[1])>0,'coherent vertical background switched'
        pot=c.PathPotential(model,sol,tau,dx,nt)
        trace,checks=c.probability(pot,tau,dt=dt)
        pv=float(trace.p_vertical.iloc[-1])
        row={'tau_s':tau,'omega_rad_s':c.np.pi/(2*tau),'temperature_K':300,
             'p_vertical_final':pv,'dx_um':dx,'dt_s':dt,'nt':nt,**checks}
        records.append(row);print(row,flush=True)
        c.pd.DataFrame(records).to_csv(c.OUT/'boundary_evaluations.csv',index=False)
        return pv
    # Bracket with the same physical parameters; no empirical boundary as input.
    low,high=15.,60.
    flo,fhi=evaluate(low)-.01,evaluate(high)-.01
    while flo*fhi>0 and high<480:
        high*=2;fhi=evaluate(high)-.01
    if flo*fhi>0:
        (c.OUT/'boundary_status.json').write_text(json.dumps({'status':'no 1% bracket up to tau=480 s'},indent=2)+'\n');return
    tau=brentq(lambda t:evaluate(t)-.01,low,high,xtol=.02)
    checks=[{'dx_um':.005,'dt_s':.005,'nt':901,'p_vertical_final':evaluate(tau)},
            {'dx_um':.0025,'dt_s':.0025,'nt':1801,'p_vertical_final':evaluate(tau,.0025,.0025,1801)}]
    result={'tau_s':tau,'omega_rad_s':c.np.pi/(2*tau),'temperature_K':300,
            'target_p_vertical':.01,'tau_root_tolerance_s':.02,'refinement_at_root':checks,
            'model':'coupled deterministic H/V relaxation; other vertical sites follow coherent positive-well path'}
    (c.OUT/'warm_boundary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BOUNDARY',result,flush=True)

if __name__=='__main__':main()
