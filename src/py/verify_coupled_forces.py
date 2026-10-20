"""Compare reduced forces to the bundled magcolloids LAMMPS, without MD steps."""
from pathlib import Path
import re,subprocess,json,hashlib
import coupled_relaxation_analysis as c
ROOT=c.p.REPO.parent
SOURCE=ROOT/'repo/data/smoke-tests/0.1-k-0.1s/lammps/job-local-serhn9wi/10/0.1/20mT/realization-1/trj1.lmpin'
EXE=ROOT/'icenumerics/icenumerics/magcolloids/magcolloids/lammps_executables/lmp_serial'
WORK=c.OUT/'static-force-audit'

def main():
    WORK.mkdir(parents=True,exist_ok=True)
    model=c.CoupledModel(nq=41)
    original=SOURCE.read_text()
    data_source=Path(re.search(r'^read_data (.*)$',original,re.M)[1])
    records=[]
    for case,(sh,sv,theta) in enumerate([(5,5,45),(2.3,4.2,47),(-1.7,3.5,55),(5.02,5.01,0)]):
        positions=model.positions([sh,sv]);lines=data_source.read_text().splitlines();start=lines.index('Atoms')+2
        for k in range(200):
            tokens=lines[start+k].split();assert int(tokens[0])==k+1
            tokens[2:5]=[f'{x:.16g}' for x in positions[k]];lines[start+k]=' '.join(tokens)
        data=WORK/f'case{case}.lmpdata';data.write_text('\n'.join(lines)+'\n')
        dump=WORK/f'case{case}.dump';script=WORK/f'case{case}.in'
        text=re.sub(r'^log .*$',f'log {WORK/f"case{case}.log"}',original,flags=re.M)
        text=re.sub(r'^read_data .*$',f'read_data {data}',text,flags=re.M)
        text=re.sub(r'^variable fieldx .*$',f'variable fieldx atom v_Bmag*sin({theta}*PI/180)',text,flags=re.M)
        text=re.sub(r'^variable fieldz .*$',f'variable fieldz atom v_Bmag*cos({theta}*PI/180)',text,flags=re.M)
        text=re.sub(r'^dump .*$','',text,flags=re.M)
        # setdipole initializes moments after the first force evaluation.
        text=re.sub(r'^run .*$',f'run 0\nrun 0\nwrite_dump Atoms custom {dump} id x y z fx fy fz mux muy muz modify sort id format float %.16g',text,flags=re.M)
        script.write_text(text)
        run=subprocess.run([str(EXE),'-in',str(script),'-screen','none','-nocite'],capture_output=True,text=True)
        assert run.returncode==0,(run.stdout,run.stderr,(WORK/f'case{case}.log').read_text()[-1000:])
        d=c.pd.read_csv(dump,sep=r'\s+',skiprows=9,header=None,names=['id','x','y','z','fx','fy','fz','mux','muy','muz'])
        force=d[['fx','fy','fz']].to_numpy()*1000 # micro force to pN
        measured=c.np.array([(force*model.uh).sum()/100,(force*model.uv).sum()/100])
        predicted=model.basis(sh,sv)@c.np.array([1,c.np.sin(c.np.deg2rad(theta))**2])+c.trap_force(c.np.array([sh,sv]))
        err=float(c.np.max(abs(measured-predicted)))
        print(case,measured,predicted,'max error',err,flush=True)
        assert err<2e-6
        records.append({'s_h_um':sh,'s_v_um':sv,'theta_deg':theta,'max_force_error_pN':err,
                        'lammps_h_pN':measured[0],'reduced_h_pN':predicted[0],
                        'lammps_v_pN':measured[1],'reduced_v_pN':predicted[1]})
    c.pd.DataFrame(records).to_csv(c.OUT/'static_force_checks.csv',index=False)
    sources=[SOURCE,data_source,EXE,ROOT/'icenumerics/icenumerics/magcolloids/magcolloids/parameters.py',Path(c.__file__),Path(__file__)]
    (c.OUT/'interaction_audit.json').write_text(json.dumps({'cutoff_um':200,'periodic_box_xy_um':300,
        'dimension':2,'images':'all in-plane lattice images inside rc; no z image layers',
        'lj_cutoff':0,'pair_style':'lj/cut/dipole/cut','field_conversion_factor':c.COEFFICIENT_MD/c.p.COEFFICIENT,
        'force_checks':'two run 0 evaluations; no MD timestep advanced','trap_branch':'outer at abs(s)>=5 um',
        'sources_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}},indent=2)+'\n')

if __name__=='__main__':main()
