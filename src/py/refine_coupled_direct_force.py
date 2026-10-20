import sys,json,time
sys.path.insert(0,'repo/src/py')
import coupled_relaxation_analysis as c
m=c.CoupledModel(nq=41)
tau=8
start=time.time()
def rhs(t,q):
 return (m.basis(*q)@c.np.array([1,c.np.sin(c.np.pi*t/(2*tau))**2])+c.trap_force(q))/c.h.GAMMA
sol=c.solve_ivp(rhs,[0,tau],[5,5],method='Radau',rtol=1e-7,atol=1e-9,max_step=tau/300,dense_output=True)
print('direct ODE',time.time()-start,sol.success,sol.nfev,flush=True)
t=c.np.linspace(0,tau,1001);q=sol.sol(t)
print('maxRv',c.np.max(abs(q[1]-5)/10),'h_cross',90*t[c.np.flatnonzero(q[0]<0)[0]]/tau,flush=True)
pot=c.PathPotential(m,sol,tau,dx=.0025,nt=1801)
trace,checks=c.probability(pot,tau,dt=.0025)
result={'tau_s':tau,'temperature_K':300,'method':'direct cutoff pair-force ODE; fine tagged probability mesh',
        'dx_um':.0025,'dt_s':.0025,'nt':1801,'p_vertical_final':float(trace.p_vertical.iloc[-1]),'maximum_Rv_d':float(c.np.max(abs(q[1]-5)/10)),**checks}
(c.OUT/'direct_force_refinement.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
