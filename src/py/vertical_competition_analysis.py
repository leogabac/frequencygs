"""Finite-volume Smoluchowski evolution along a computed coherent horizontal path.

No MD runs or fitted rate factors. Every output is separate from empirical tables.
"""
from pathlib import Path
import os, json, hashlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/frequencygs-matplotlib')
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.special import exprel
from scipy.optimize import brentq
import rotating_potential_analysis as p
import collective_horizontal_analysis as h
OUT=p.REPO/'data/figures/vertical-competition'

class VerticalPotential:
    def __init__(self, dx=.01, extent=5.2, nq=161):
        self.s=np.arange(-extent,extent+dx/2,dx)
        self.dx=dx
        state=pd.read_csv(p.STATE)
        self.state=state
        self.tag=int(state.loc[state.dy.abs()>.5,'id'].iloc[0])
        q=np.linspace(-5.1,5.1,nq)
        a,x=[],[]
        for sh in q:
            positions=p.background(state,(5-sh)/10)
            aa,xx=p.geometry(state,positions,self.tag,s=self.s)
            a.append(aa);x.append(xx)
        self.A=RectBivariateSpline(q,self.s,np.array(a))
        self.X=RectBivariateSpline(q,self.s,np.array(x))
    def energy(self, sh, theta):
        return p.trap_potential(self.s)+p.COEFFICIENT*(self.A(sh,self.s)[0]-3*np.sin(theta)**2*self.X(sh,self.s)[0])
    def force(self,s,sh,theta):
        magnetic=-p.COEFFICIENT*(self.A(sh,s,dy=1)[0,0]-3*np.sin(theta)**2*self.X(sh,s,dy=1)[0,0])
        trap=-np.where(abs(s)<=5,-8*p.HEIGHT*s/100,p.STIFFNESS*(s-5*np.sign(s)))
        return magnetic+trap

def horizontal_solution(model,tau):
    return solve_ivp(lambda t,y:[-model.gradient(y[0],90*t/tau)/h.GAMMA],
                     [0,tau],[5.],method='Radau',rtol=1e-9,atol=1e-10,max_step=tau/200,dense_output=True)

def rates(u,temperature,dx):
    z=np.diff(u)/(p.KB*temperature)
    # Bernoulli B(z)=z/(exp(z)-1). These rates satisfy detailed balance exactly.
    def bernoulli(v):
        out=np.zeros_like(v)
        small=np.abs(v)<50
        out[small]=1/exprel(v[small])
        pos=v>=50
        out[pos]=v[pos]*np.exp(-v[pos])/(1-np.exp(-v[pos]))
        neg=v<=-50
        out[neg]=-v[neg]/(1-np.exp(v[neg]))
        return out
    factor=p.KB*temperature/h.GAMMA/dx**2
    return factor*bernoulli(z),factor*bernoulli(-z)

def integrate(potential,sol,tau,temperature,dt=.01):
    s,dx=potential.s,potential.dx
    mass=np.zeros(len(s));mass[np.argmin(abs(s-5))]=1
    tend=tau-.05
    n=int(np.ceil(tend/dt));step=tend/n
    samples=[];minmass=1.;maxerr=0.
    band=np.zeros((3,len(s)))
    for k in range(n):
        t=(k+1)*step
        sh=float(sol.sol(t)[0]);theta=np.pi*t/(2*tau)
        u=potential.energy(sh,theta)
        forward,backward=rates(u,temperature,dx)
        band[1]=1
        band[1,:-1]+=step*forward;band[1,1:]+=step*backward
        band[0,1:]=-step*backward;band[2,:-1]=-step*forward
        mass=solve_banded((1,1),band,mass,check_finite=False)
        if k%max(1,int(.05/step))==0 or k==n-1:
            # A cell centered on zero contributes half its mass to each side.
            pv=mass[s<-dx/4].sum()+.5*mass[abs(s)<dx/4].sum()
            samples.append([t,np.rad2deg(theta),sh,pv,float(mass@s)])
            minmass=min(minmass,mass.min());maxerr=max(maxerr,abs(mass.sum()-1))
    assert minmass>=-1e-13 and maxerr<1e-8,(minmass,maxerr)
    outer=mass[(s<-5.15)|(s>5.15)].sum()
    return pd.DataFrame(samples,columns=['t_s','theta_deg','s_h_um','p_vertical','mean_s_v_um']),mass,{'mass_error':maxerr,'minimum_mass':minmass,'outer_wall_mass':float(outer)}

def verification():
    # Independent equilibrium identity; evolution checks conservation separately.
    s=np.linspace(-1,1,201);u=.02*s*s
    f,b=rates(u,300,.01)
    eq=np.exp(-(u-u.min())/(p.KB*300));eq/=eq.sum()
    assert np.max(abs(f*eq[:-1]-b*eq[1:]))<1e-12
    return {'detailed_balance_max_flux':float(np.max(abs(f*eq[:-1]-b*eq[1:])))}

def warm_boundary(pot,model,dt):
    records=[]
    def objective(tau):
        sol=horizontal_solution(model,tau)
        assert sol.success
        trace,mass,checks=integrate(pot,sol,tau,300,dt)
        pv=float(trace.p_vertical.iloc[-1])
        records.append({'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'temperature_K':300,
                        'p_vertical_final':pv,'dx_um':pot.dx,'dt_s':dt,**checks})
        print('Warm boundary evaluation:',tau,pv,flush=True)
        return pv-.01
    tau=brentq(objective,14,15,xtol=.002)
    pd.DataFrame(records).to_csv(OUT/'warm_boundary_root_evaluations.csv',index=False)
    result={'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'temperature_K':300,
            'target_p_vertical':.01,'dx_um':pot.dx,'dt_s':dt,'root_tau_tolerance_s':.002}
    (OUT/'warm_boundary_root.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--dx',type=float,default=.01);parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--warm-boundary',action='store_true')
    parser.add_argument('--taus',type=float,nargs='+',default=[7,8,9,10,20,30,40]);parser.add_argument('--temperatures',type=float,nargs='+',default=[.1,300])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    pot=VerticalPotential(args.dx);model=h.CollectivePath();rows=[];traces=[]
    print('Geometry ready; checks:',verification(),flush=True)
    for tau in args.taus:
        sol=horizontal_solution(model,tau);assert sol.success
        for temp in args.temperatures:
            trace,mass,check=integrate(pot,sol,tau,temp,args.dt)
            trace['tau_s']=tau;trace['temperature_K']=temp;traces.append(trace)
            row={'tau_s':tau,'omega_rad_s':np.pi/(2*tau),'temperature_K':temp,'p_vertical_final':trace.p_vertical.iloc[-1],'dx_um':args.dx,'dt_s':args.dt,**check}
            rows.append(row);print(row,flush=True)
    suffix=f'dx{args.dx:g}-dt{args.dt:g}'
    pd.DataFrame(rows).to_csv(OUT/f'probabilities-{suffix}.csv',index=False)
    pd.concat(traces).to_csv(OUT/f'probability_traces-{suffix}.csv',index=False)
    if args.warm_boundary:
        print(warm_boundary(pot,model,args.dt),flush=True)
    metadata={'model':'tagged vertical Smoluchowski equation on deterministic coherent horizontal path',
              'parameters_verified':p.verify_parameter_quantities(),'vertical_id':pot.tag,
              'noise_temperatures_K':args.temperatures,'gamma_pN_s_per_um':h.GAMMA,
              'geometry':'nearest periodic image; no hard cutoff; all other verticals frozen',
              'solver':'exponentially fitted finite volume; backward Euler; no-flux outer boundaries',
              'initial_condition':'unit point mass at s=+5 um','observation_time':'tau-0.05 s',
              'probability':'instantaneous mass at s<0; half central-cell mass; includes returns',
              'domain_um':[-5.2,5.2],'checks':verification(),
              'sources_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                  [Path(__file__),Path(p.__file__),Path(h.__file__),p.STATE,p.REPO/'src/parameters.py']}}
    (OUT/f'provenance-{suffix}.json').write_text(json.dumps(metadata,indent=2)+'\n')
