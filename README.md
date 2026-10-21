# Transition between ground states in square anisotropic artificial colloidal ice

Simulation and analysis code for the paper by **Leonardo G. Alanis-Cantú** and
**Antonio Ortiz-Ambriz**.

We study how rotating an external magnetic field drives square anisotropic
artificial colloidal ice from an ice-rule ground state to a charged crystal.
The simulations examine how the transformation depends on rotation rate,
field strength, and temperature.

## Paper figures

The current figure notebooks are in [`src/notebooks`](src/notebooks). They load
saved analysis tables and generate the paper plots without running simulations.

| Notebook | Output |
| --- | --- |
| [`figure2.ipynb`](src/notebooks/figure2.ipynb) | Vertex populations during field rotation at 10 and 20 mT |
| [`figure3.ipynb`](src/notebooks/figure3.ipynb) | Final vertex populations and charge order, including the temperature comparison |
| [`figure4.ipynb`](src/notebooks/figure4.ipynb) | **Figures 4 and 5:** ensemble and individual-particle displacements, switching activity, and switching-width analysis |
| [`figure6.ipynb`](src/notebooks/figure6.ipynb) | Field–rotation-rate diagram and temperature-dependent transition inset |
| [`extra_figures.ipynb`](src/notebooks/extra_figures.ipynb) | Zero-field autocorrelation, thermal decorrelation time, and endpoint observables |

Each notebook contains notes on the observables, source data, and plotting
choices. Other notebooks in this directory contain earlier or exploratory
analyses.

### Data and plotting bundle

The plotting inputs are stored under `data/test11/10/` and
`data/figures/paper/`. **The `data/` directory is excluded from Git**, so a clone
alone does not contain the inputs needed to render the figures.

The accompanying **`frequencygs-plot-data.tar.gz`** bundle contains the required
summary tables, the cached individual-particle traces for Figure 4, clean
copies of the figure notebooks, and the Figure 5 fitting helper. It also
includes source metadata, a manifest, and SHA256 checksums. It does not contain
raw simulation trajectories or generated figures.

If you have the bundle, extract it with:

```bash
tar -xzf frequencygs-plot-data.tar.gz
cd frequencygs-plot-data
sha256sum -c SHA256SUMS
```

The extracted directory can be used independently of this checkout. Keep its
folder layout intact. If using a clone instead, copy the bundle's `data/`
directory into the repository root.

### Run the notebooks

A Python environment with NumPy, pandas, SciPy, Matplotlib, IPython, and Jupyter
is sufficient for these figure notebooks. For example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy matplotlib jupyterlab
jupyter lab src/notebooks
```

The plots use Matplotlib's LaTeX text rendering, so a working LaTeX installation
and its Matplotlib font packages are also required. Run each notebook from top
to bottom. Figures are saved as PDF and PNG under `data/figures/paper/`, with
supporting figures in its `extra/` subdirectory.

The plotting notebooks use saved inputs; they do not require LAMMPS,
`icenumerics`, or access to the original simulation disk.

## Simulations and analysis

To obtain the code:

```bash
git clone https://github.com/leogabac/frequencygs.git
cd frequencygs
```

The main simulation drivers are:

- [`test11.py`](src/py/test11.py): the historical paper baseline.
- [`test11_improved.py`](src/py/test11_improved.py): the separate driver for newer
  campaigns, with isolated outputs, run manifests, and Parquet trajectory storage.

Simulation setup and command-line examples are documented in
[`src/py/README.md`](src/py/README.md). Simulations additionally require the
project's [`icenumerics` fork](https://github.com/leogabac/icenumerics) and its
LAMMPS-based simulation environment. The historical setup uses the fork's
`dev` branch in a sibling directory. [`runtime-requirements.txt`](runtime-requirements.txt)
lists pinned Python runtime dependencies; [`requirements.txt`](requirements.txt)
contains the broader historical notebook environment.

The repository also contains:

- `src/auxnumerics/`: auxiliary analysis routines.
- `src/parameters.py` and related parameter files: simulation parameters.
- `src/py/`: simulation drivers, data reduction, campaign tools, and exploratory
  mechanism analyses. Not every exploratory analysis is used in the paper.
- `jobs/`: batch-job scripts.

Historical simulation outputs and newer campaigns are kept separate. The figure
notebooks and their provenance files identify which saved results enter each plot.

## License

GNU General Public License, version 2. See [`LICENSE`](LICENSE).
