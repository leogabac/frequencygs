import sys,json
sys.path.insert(0,'repo/src/py')
import coupled_relaxation_analysis as c
m=c.CoupledModel()
checks=[]
for tau in [8,15]:
 sol=m.solve(tau);tight=m.solve(tau,tol=1e-10)
 t=c.np.linspace(0,tau,101);q=sol.sol(t)
 errors=[]
 for k in range(len(t)):
  w=c.np.sin(c.np.pi*t[k]/(2*tau))**2
  force=m.basis(*q[:,k])@c.np.array([1,w])+c.trap_force(q[:,k])
  errors.append(c.np.max(abs(force-m.force(q[:,k],c.np.pi*t[k]/(2*tau)))))
 checks.append({'tau_s':tau,'tighter_ode_max_position_difference_um':float(c.np.max(abs(sol.sol(t)-tight.sol(t)))),
                'force_spline_vs_direct_max_error_pN':float(max(errors)),
                'horizontal_crossing_angle_deg':90*sol.t_events[0][0]/tau,
                'vertical_min_um':float(q[1].min())})
print(checks,flush=True)
(c.OUT/'numerical_force_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
