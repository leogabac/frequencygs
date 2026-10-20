# Python Scripts
In this directory, you can find regular python scripts for either
1. Simulations.
2. Long-running computations for later visualization.

## Simulations

- **test11_improved.py**\
The driver for new campaign development. It starts from the repaired
`test11.py` pipeline and reuses `test11_field.py`. Its default campaign
directory is `DATA_ROOT/test11_improved`; `test11.py` remains the baseline.
The improved driver stores `FIELD/trj/trjN.parquet`, `FIELD/xtrjN.parquet`,
and `FIELD/verticesN.parquet` using Zstandard compression, preserving numeric
precision and explicit `frame`/`id` or `frame`/`vertex` columns. LAMMPS dumps
are converted directly into Parquet row groups without intermediate CSV files.
Kappa and vertex-fraction summaries remain CSV. Readers prefer Parquet but
accept historical CSV input; original data is not migrated or deleted.

Install the added Parquet dependency in the simulation environment:
```bash
python -m pip install pyarrow==20.0.0
```
The improved driver checks this dependency before starting simulations.
Fractional rotation times are supported. `--fields`, `--realizations`, and
`--framerate` allow small smoke runs without editing `parameters.py`:
```bash
python test11_improved.py 10 0.1 --temperature 0.1 --fields 20 --realizations 2 --framerate 100 --data-root /path/to/smoke-results
```
The bundled LAMMPS 18 Jun 2019 `fix bd` rejects zero temperature. A 0.1 K,
0.1-second run with two realizations at 20 mT completed the full Parquet and
analysis pipeline. With historical endpoint removal, its 11 raw frames become
10 cleaned frames.
The baseline `test11.py` continues to write CSV tables. The standalone historical
analysis scripts still read CSV; use the improved driver's `-v -k -a` stages
for the Parquet pipeline.

- **run_local_campaign.py**\
Runs independent rotation times without PBS. Defaults: times 1 through 10 s,
four simultaneous drivers, ten realizations each, size 10, 0.1 K, 20 frames/s,
and fields 0 through 20 mT. Install `runtime-requirements.txt` into a dedicated
environment, then run:
```bash
.venv/bin/python -u src/py/run_local_campaign.py --name cold-tau1-10
```
Results go into `repo/data/campaigns/NAME`. Each tau gets a stdout log and
`resources.json` with elapsed wall time, return code, and sampled peak summed
RSS across its Python process and LAMMPS children. Shared memory pages may be
counted more than once. Numerical-library threads are limited to one per
process. The launcher refuses to reuse an existing campaign directory.
For a detached run that continues after SSH disconnects:
```bash
bash src/py/start_local_campaign.sh cold-tau1-10
```
Its launcher log and PID file are stored beside the campaign directory.

- **submit_temperature_campaign.py**\
Submits the 1 K campaign to the `batch` queue on `ts-softmatter02`, using
`jobs/temperature-array.pbs`. The schedule is tau 1–20, 30, 40, 50, 70, 80,
90, and 100 s (27 jobs). Each job requests ten CPUs, 2 GiB, and 48 hours;
the PBS array permits six simultaneous jobs, scheduling the longest times first.
A ten-realization, 0.1 s pilot at 20 mT and 1 K runs first. The main array
depends on its successful completion. Particle drag remains at the original
reference value; only the integrator noise temperature changes.
```bash
.venv/bin/python src/py/submit_temperature_campaign.py --name softmatter02-1K-tau1-100
```
Campaign settings, submitted job IDs, the environment, PBS output, simulation
manifests, and resource measurements are persisted under `data/campaigns/NAME`.
Each tau owns its trajectories and summary files. The 1000 s run is separate.

- **test11.py**\
Simulation at selected $\omega = \pi/2/\tau$ and magnetic $B$. For more information, run
```
python test11.py --help
```
Raw LAMMPS files are isolated by invocation, PBS job ID (when available),
size, rotation time, field, and realization. The script prints the raw output
directory. Use `--lammps-root /path/to/scratch` to select its root; the default
is `src/py/lammps_files`. Each invocation creates a unique directory, including
when the same PBS job launches the script more than once. Converted CSV output
still uses the campaign data path defined in `test11.py`.
Before launching each concurrent realization pack, the script writes one
`manifest.json` in its raw field directory and copies it to the campaign field
directory with a unique invocation name. It records parameters with units,
resolved particle/trap properties, field expressions, output settings, PBS job
ID, initial-state hash, and each realization's seeds and generated input hashes.

With no stage flags, `test11.py` runs simulations, converts dumps into
`FIELD/trj/trjN.csv`, cleans into `FIELD/xtrjN.csv`, computes every frame's
vertices into `FIELD/verticesN.csv`, and computes kappa and vertex fractions.
Cleaning preserves the historical final-frame removal when magnetic moments
are present. Vertex outputs include physical timestamps; kappa is computed
from their charges rather than recalculating topology.

For example, from any working directory:
```bash
python /path/to/repo/src/py/test11.py 10 100 --temperature 0 --data-root /path/to/cold-results --lammps-root /path/to/scratch
```
The noise temperature defaults to the existing value in `parameters.py`.
Changing it does not change the particle drag derived from the reference
temperature and diffusion coefficient. Zero-temperature support in the bundled
LAMMPS integrator still needs a smoke run.

Each rotation time owns `SIZE/TIME/kappa_all.csv` and
`SIZE/TIME/average_counts.csv`, avoiding shared size-level appends across PBS
jobs. Existing historical size-level summaries are left untouched. Existing
trajectory outputs are protected against overwriting; use a fresh data root
for a new campaign. Explicit stage flags run only the selected stages:
`-s` simulation/conversion, `-v` cleaning/vertices, `-k` kappa, and `-a` vertex
fractions. For postprocessing only, use `-v -k -a`; `-k` and `-a` require
vertex files with timestamps generated by this pipeline.

- **test11_field.py** Magnetic field string for LAMMPS.

## Long-Running Jobs

- **make_chir.py**  
Computes the chirality order parameter for all frames in bulk. All $\omega$, $B$ at given size.

- **make_kappa.py**  
Computes the charge order parameter for all frames in bulk. All $\omega$, $B$ at given size.

- **take_distances.py**  
Computes the travelled distance (parallel) at any time $\mathbf{r}(t) = | \mathbf{r}(t) - \mathbf{r}(0) |$ in bulk.  
Schema:
| 1 ... N | total_time | field | realization | t | theta

- **trim_distances.py**  
Files from `take_distances.py` are too big to handle. This script trims a lighter version for 20mT only.
