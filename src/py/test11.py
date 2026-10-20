# ====================
# Test 10
# Start from an ordered state
# z->x
# ====================

import os
import sys
from pathlib import Path
import tempfile
import json
import hashlib
import shutil
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT.parent / 'icenumerics'))
sys.path.insert(0, str(SCRIPT_DIR.parent / 'auxnumerics'))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import icenumerics as ice
import concurrent.futures
import auxiliary as aux
import vertices as vrt
from campaign_analysis import clean_trajectory, compute_vertices, analyze_vertices, save_csv

from parameters import params
from tqdm import tqdm
import importlib
import argparse

ureg = ice.ureg
idx = pd.IndexSlice


def create_simulation(params, trj, size, realization):

    global fx
    global fy
    global fz

    params = params.copy()
    params['size'] = size
    N = size
    a = params['lattice_constant']

    print('time sanity check: ', params['total_time'])

    sp = ice.spins()
    sp.create_lattice('square', [N, N], lattice_constant=a, border='periodic')

    particle = ice.particle(radius=params['particle_radius'],
                            susceptibility=params['particle_susceptibility'],
                            diffusion=params['particle_diffusion'],
                            temperature=params['particle_temperature'],
                            density=params['particle_density'])

    trap = ice.trap(trap_sep=params['trap_sep'],
                    height=params['trap_height'],
                    stiffness=params['trap_stiffness'])

    params['particle'] = particle
    params['trap'] = trap

    col = aux.trj2col(params, trj)

    world = ice.world(
        field=params['max_field'],
        temperature=params['sim_temp'],
        dipole_cutoff=params['sim_dipole_cutoff'],
        boundaries=['p', 'p', 'p']
    )

    col.simulation(world,
                   name=f'trj{realization}',
                   include_timestamp=False,
                   targetdir=str(Path(params['lammps_dir']) / f'realization-{realization}'),
                   framerate=params['framespersec'],
                   timestep=params['dt'],
                   run_time=params['total_time'],
                   output=['x', 'y', 'z', 'mux', 'muy', 'muz'],
                   processors=1)

    col.sim.field.fieldx = "".join(fx)
    col.sim.field.fieldy = "".join(fy)
    col.sim.field.fieldz = "".join(fz)

    return col


def manifest_value(value):
    if hasattr(value, 'magnitude') and hasattr(value, 'units'):
        return {'value': manifest_value(value.magnitude), 'unit': str(value.units)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return manifest_value(value.tolist())
    if isinstance(value, np.generic):
        return manifest_value(value.item())
    if isinstance(value, dict):
        return {key: manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [manifest_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported manifest value: {type(value)}')


def manifest_record(col, params, size, realization):
    sim = col.sim
    source = Path(params['initial_state_path'])
    files = {key: str(Path(getattr(sim, key)).resolve()) for key in
             ('script_name', 'input_name', 'output_name', 'log_name')}
    manifest = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'pbs_job_id': os.environ.get('PBS_JOBID'),
        'command': sys.argv,
        'size': size,
        'realization': realization,
        'parameters': {**params, 'size': size},
        'initial_state': {'path': str(source),
                          'sha256': params['initial_state_sha256']},
        'particle': vars(col[0].particle),
        'trap': vars(col[0].trap),
        'height_spread': col.height_spread,
        'susceptibility_spread': col.susceptibility_spread,
        'region': col.region,
        'boundaries': sim.world.boundaries,
        'integrator': sim.world.integrator_def,
        # this wrapper embeds the world's seed in the actual integrator string
        'integrator_seed': sim.world.seed,
        'simulation_seed': sim.seed,
        'field_expressions': {axis: getattr(sim.field, 'field' + axis)
                              for axis in ('x', 'y', 'z')},
        'field_variables': sim.field.variable_def,
        'run': {'timestep': sim.timestep, 'framerate': sim.framerate,
                'total_time': sim.total_time, 'output': sim.output,
                'processors': sim.processors},
        'files': files,
        'generated_sha256': {key: hashlib.sha256(Path(files[key]).read_bytes()).hexdigest()
                             for key in ('script_name', 'input_name')},
    }
    return manifest


def write_manifest(cols, params, size, realizations):
    manifest = {'schema_version': 1, 'realizations': [
        manifest_record(col, params, size, realization)
        for col, realization in zip(cols, realizations)]}
    manifest['postprocessing'] = {
        'remove_last_frame_when_moments_present': True,
        'vertex_frame_stride': 1,
        'kappa_definition': 'sum(q[i,j] * (-1)**(i+j)); signed, unnormalized',
    }
    path = Path(params['lammps_dir']) / 'manifest.json'
    with path.open('x') as stream:
        json.dump(manifest_value(manifest), stream, indent=2, allow_nan=False)
        stream.write('\n')


def run_simulation(col):
    result = col.sim.run()
    if result.returncode:
        raise RuntimeError(f"LAMMPS failed: {col.sim.script_name}\n"
                           f"{result.stdout.decode(errors='replace')}\n"
                           f"{result.stderr.decode(errors='replace')}")


def load_simulation(params, trj, data_path, size, realization):

    print(f'Saving {realization}...')
    col = create_simulation(params, trj, size, realization)
    col.sim.base_name = os.path.join(col.sim.dir_name, col.sim.file_name)
    col.sim.script_name = col.sim.base_name+'.lmpin'
    col.sim.input_name = col.sim.base_name+'.lmpdata'
    col.sim.output_name = col.sim.base_name+'.lammpstrj'
    col.sim.log_name = col.sim.base_name+'.log'
    trj_path = os.path.join(data_path, 'trj')

    try:
        os.mkdir(trj_path)
    except:
        pass

    ice.get_ice_trj_low_memory(col, dir_name=trj_path)

# ===== MAIN ROUTINE ===== #


parser = argparse.ArgumentParser(description="Run a rotation z -> x for many different times")

# flags
parser.add_argument('-s', '--sims', action='store_true', help='run simulations')
parser.add_argument('-v', '--vertices', action='store_true', help='run vertices')
parser.add_argument('-a', '--averages', action='store_true', help='run vertices averages')
parser.add_argument('-k', '--kappa', action='store_true', help='compute kappa from vertex charges')
parser.add_argument('--data-root', type=Path, default=REPO_ROOT / 'data',
                    help='campaign data root; use a separate root for the cold campaign')
parser.add_argument('--temperature', type=float, default=params['sim_temp'].to(ureg.K).magnitude,
                    help='noise temperature in kelvin; particle drag remains unchanged')
parser.add_argument('--lammps-root', type=Path,
                    default=Path(__file__).resolve().parent / 'lammps_files',
                    help='root for isolated raw LAMMPS files (may be PBS scratch)')

# positional arguments
parser.add_argument('size', type=str, help='The size input')
parser.add_argument('time', type=str, help='The size input')

args = parser.parse_args()

sth_passed = any([args.sims, args.vertices, args.averages, args.kappa])
if not sth_passed:
    args.sims = True
    args.vertices = True
    args.averages = True
    args.kappa = True

# importing the field
script_name = Path(__file__).stem
module_name = f'{script_name}_field'

if int(args.size) <= 0 or int(args.time) <= 0 or not np.isfinite(args.temperature) or args.temperature < 0:
    parser.error('size/time must be positive and temperature must be finite and nonnegative')
params['total_time'] = int(args.time) * ureg.s
params['sim_temp'] = args.temperature * ureg.K

module = importlib.import_module(module_name)
fx = module.fx
fy = module.fy
fz = module.fz

print(fx)
print(fy)
print(fz)

SIZE = args.size
REALIZATIONS = list(range(1, 11))
FIELDS = list(range(21))

DATA_PATH = args.data_root.resolve() / script_name
SIZE_PATH = os.path.join(DATA_PATH, str(SIZE))
if args.sims:
    params['initial_state_path'] = REPO_ROOT / 'data' / 'states' / 'ice' / f'{SIZE}.csv'
    initial_state_bytes = params['initial_state_path'].read_bytes()
    params['initial_state_sha256'] = hashlib.sha256(initial_state_bytes).hexdigest()
    GSTRJ = pd.read_csv(io.BytesIO(initial_state_bytes), index_col='id')

if not os.path.isdir(SIZE_PATH):
    os.makedirs(SIZE_PATH)


if args.sims:
    args.lammps_root = args.lammps_root.resolve()
    args.lammps_root.mkdir(parents=True, exist_ok=True)
    job_label = os.environ.get('PBS_JOBID', 'local').replace('/', '_')
    run_dir = Path(tempfile.mkdtemp(prefix=f'job-{job_label}-', dir=args.lammps_root))
    print(f'Raw LAMMPS files: {run_dir}', flush=True)
    print('RUNNING SIMULATIONS')
    for B_mag in FIELDS:
        os.system('clear')

        params["max_field"] = B_mag * ureg.mT
        params['lammps_dir'] = run_dir / str(SIZE) / str(module.rtime) / f'{B_mag}mT'
        print('Time: ', params['total_time'])
        print('Field:', params["max_field"])
        field_path = os.path.join(
            SIZE_PATH, str(module.rtime), str(B_mag) + 'mT')
        if not os.path.isdir(field_path):
            os.makedirs(field_path)

        if any(Path(field_path).glob('trj/trj*.csv')) or any(Path(field_path).glob('trj*.csv')):
            raise FileExistsError(f'Refusing to overwrite trajectories in {field_path}; choose another --data-root')

        cols = [create_simulation(params, GSTRJ, int(SIZE), i) for i in REALIZATIONS]
        for col in cols:
            col.sim.generate_scripts()
        write_manifest(cols, params, int(SIZE), REALIZATIONS)
        shutil.copy2(Path(params['lammps_dir']) / 'manifest.json',
                     Path(field_path) / f'{run_dir.name}.manifest.json')

        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            results = list(
                executor.map(
                    run_simulation,
                    cols
                )
            )
        for i in REALIZATIONS:
            print(f'===== Realization {i} =====')
            load_simulation(params, GSTRJ, field_path, int(SIZE), i)

# Postprocess after all LAMMPS packs have finished and been converted.
timepath = Path(SIZE_PATH) / str(module.rtime)
field_paths = sorted((p for p in timepath.glob('*mT') if p.is_dir()),
                     key=lambda p: int(p.name[:-2]))
if args.vertices or args.kappa or args.averages:
    if not field_paths:
        raise FileNotFoundError(f'No field directories in {timepath}')
if args.vertices:
    for field_path in field_paths:
        for realization in REALIZATIONS:
            print(f'Cleaning and vertices: {field_path}, realization {realization}', flush=True)
            clean_trajectory(field_path, realization)
            compute_vertices(field_path, realization, int(SIZE),
                             params['lattice_constant'].to(ureg.um).magnitude)
if args.kappa or args.averages:
    kappa, averages = analyze_vertices(field_paths, REALIZATIONS, int(SIZE),
                                      module.rtime, kappa=args.kappa, averages=args.averages)
    # each omega job owns its summaries; no concurrent appends to size-level files
    if args.kappa:
        save_csv(kappa, timepath / 'kappa_all.csv')
    if args.averages:
        save_csv(averages, timepath / 'average_counts.csv')
