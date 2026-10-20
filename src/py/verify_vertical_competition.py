import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'repo/src/py')
import vertical_competition_analysis as v
st=v.pd.read_csv(v.p.STATE);ids=st.loc[st.dy.abs()>.5,'id'].to_numpy()
errors=[]
for alpha in [0,.3,.5,.8,1]:
 pos=v.p.background(st,alpha)
 for which in [0,1]:
  profiles=[]
  for tag in ids:
   a,x=v.p.geometry(st,pos,tag,s=v.np.linspace(-5,5,7));profiles.append([a,x][which])
  errors.append(float(v.np.ptp(v.np.array(profiles),axis=0).max()))
assert max(errors)<1e-15
pot=v.VerticalPotential(.005);sh=1.234;s=v.np.array([-4.321,0,3.456]);pos=v.p.background(st,(5-sh)/10)
a,x=v.p.geometry(st,pos,pot.tag,s=s)
err=max(v.np.max(abs(pot.A(sh,s)[0]-a)),v.np.max(abs(pot.X(sh,s)[0]-x)))
model=v.h.CollectivePath();sol=v.horizontal_solution(model,14.67)
large=v.VerticalPotential(.005,extent=5.4)
trace,mass,check=v.integrate(large,sol,14.67,300,.0025)
results={'all_100_vertical_sites_geometrically_equivalent_max_error':max(errors),'interpolation_geometric_max_error':float(err),'expanded_domain_5p4_um_p_vertical_tau14p67':float(trace.p_vertical.iloc[-1]),'expanded_domain_checks':check}
old={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for directory in [v.p.OUTPUT,v.h.OUT] for f in directory.glob('*') if f.suffix in ['.pdf','.png']}
results['preserved_figures_sha256']=old
(v.OUT/'geometry_and_domain_checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(results,flush=True)
