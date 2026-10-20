"""Tagged-particle potential scans adapted from the StuckGS Figure 5 construction.

Coordinates: um; energy: pN um. No simulations or rate fits are performed.
The full lattice uses nearest periodic images, without a hard radial cutoff.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/frequencygs-matplotlib')
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / 'data/figures/rotating-potentials'
STATE = REPO / 'data/states/ice/10.csv'
REFERENCE = Path('/home/holo/documents/mnt/stuckgs/latex/version3/Figures/Figure5')
# Same values as src/parameters.py, in um and pN um; no icenumerics dependency.
RADIUS = 5.
CHI = .0576
FIELD_MT = 20.
SEPARATION = 10.
HEIGHT = .004
STIFFNESS = 100.
BOX = 300.
KB = 1.380649e-5  # pN um/K
REFERENCE_TEMPERATURE = 300.
MU0 = 4*np.pi*1e-7
MOMENT = 4*np.pi*(RADIUS*1e-6)**3/3 * CHI*(FIELD_MT*1e-3)/MU0
COEFFICIENT = MU0*MOMENT**2/(4*np.pi)*1e36  # pN um^4
S = np.unique(np.r_[np.linspace(-5.625, 5.625, 4501), -5., 0., 5.])
ANGLES = np.linspace(30, 90, 241)
STAGES = [0., .5, 1.]


def trap_potential(s):
    s = np.asarray(s)
    return np.where(np.abs(s) <= SEPARATION/2,
                    HEIGHT*(1-4*s*s/SEPARATION**2),
                    STIFFNESS/2*(np.abs(s)-SEPARATION/2)**2)


def background(state, alpha):
    centers = state[['x', 'y', 'z']].to_numpy()
    relative = state[['cx', 'cy', 'cz']].to_numpy().copy()
    horizontal = np.abs(state.dx.to_numpy()) > .5
    relative[horizontal] *= 1-2*alpha
    return centers+relative


def geometry(state, positions, particle_id, s=S, periodic=True):
    ids = state.id.to_numpy()
    j = int(np.flatnonzero(ids == particle_id)[0])
    center = state.loc[j, ['x', 'y', 'z']].to_numpy(dtype=float)
    axis = state.loc[j, ['dx', 'dy', 'dz']].to_numpy(dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    initial = state.loc[j, ['cx', 'cy', 'cz']].to_numpy(dtype=float)
    # Positive s is the initially occupied side for either orientation.
    if np.dot(axis, initial) < 0:
        axis *= -1
    displacement = center + np.asarray(s)[:, None, None]*axis-positions[None]
    if periodic:
        displacement -= BOX*np.rint(displacement/BOX)
    distance = np.linalg.norm(displacement, axis=2)
    distance[:, j] = np.inf
    assert np.all(distance > 0)
    A = np.sum(distance**-3, axis=1)
    X = np.sum(displacement[:, :, 0]**2*distance**-5, axis=1)
    return A, X


def profile(A, X, theta):
    return trap_potential(S)+COEFFICIENT*(A-3*np.sin(np.deg2rad(theta))**2*X)


def directed_barrier(state, positions, particle_id, theta, energy):
    minima = find_peaks(-energy)[0]
    maxima = find_peaks(energy)[0]
    occupied = minima[S[minima] > 0]
    opposite = minima[S[minima] < 0]
    record = {'occupied_minimum_um': np.nan, 'opposite_minimum_um': np.nan,
              'saddle_um': np.nan, 'barrier_pN_um': np.nan, 'barrier_kBTref': np.nan}
    if not len(occupied):
        # No metastable positive-side well: no activation barrier for leaving it.
        return {**record, 'status': 'occupied_well_absent', 'barrier_pN_um': 0., 'barrier_kBTref': 0.}
    if not len(opposite):
        return {**record, 'status': 'opposite_well_absent'}
    i = occupied[np.argmin(np.abs(S[occupied]-5))]
    j = opposite[np.argmin(np.abs(S[opposite]+5))]
    saddles = maxima[(maxima > j) & (maxima < i)]
    if not len(saddles):
        return {**record, 'status': 'no_intervening_maximum'}
    k = saddles[np.argmax(energy[saddles])]
    def evaluate(s):
        a, x = geometry(state, positions, particle_id, s=np.atleast_1d(s))
        return float(trap_potential(s)+COEFFICIENT*(a[0]-3*np.sin(np.deg2rad(theta))**2*x[0]))
    def refine(index, sign):
        candidates = [S[index]]
        result = minimize_scalar(lambda s: sign*evaluate(s),
                                 bounds=(S[index-1], S[index+1]), method='bounded',
                                 options={'xatol': 1e-10})
        candidates.append(result.x)
        for cusp in [-5., 5.]:
            if S[index-1] <= cusp <= S[index+1]:
                candidates.append(cusp)
        best = min(candidates, key=lambda s: sign*evaluate(s))
        return best, evaluate(best)
    occupied_s, occupied_u = refine(i, 1)
    opposite_s, _ = refine(j, 1)
    saddle_s, saddle_u = refine(k, -1)
    barrier = saddle_u-occupied_u
    assert barrier >= 0
    return {'status': 'two_wells', 'occupied_minimum_um': occupied_s,
            'opposite_minimum_um': opposite_s, 'saddle_um': saddle_s,
            'barrier_pN_um': barrier, 'barrier_kBTref': barrier/(KB*REFERENCE_TEMPERATURE)}


def compute():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = pd.read_csv(STATE)
    assert len(state) == 200 and state.id.is_unique
    assert np.allclose(np.linalg.norm(state[['cx', 'cy', 'cz']], axis=1), 5)
    vertical_id = int(state.loc[state.dy.abs() > .5, 'id'].iloc[0])
    horizontal_id = int(state.loc[state.dx.abs() > .5, 'id'].iloc[0])
    curves, records = [], []
    for alpha in STAGES:
        positions = background(state, alpha)
        A, X = geometry(state, positions, vertical_id)
        for theta in ANGLES:
            u = profile(A, X, theta)
            records.append({'orientation': 'vertical', 'alpha': alpha, 'theta_deg': theta,
                            **directed_barrier(state, positions, vertical_id, theta, u)})
            if theta in np.linspace(35, 60, 11):
                reference = np.interp(5., S, u)
                curves.append(pd.DataFrame({'orientation': 'vertical', 'alpha': alpha,
                                            'theta_deg': theta, 's_um': S,
                                            'U_pN_um': u, 'U_relative_kBTref': (u-reference)/(KB*REFERENCE_TEMPERATURE)}))
    # Horizontal scan is a tagged particle, not a coherent all-horizontal flip.
    positions = background(state, 0.)
    A, X = geometry(state, positions, horizontal_id)
    for theta in ANGLES:
        u = profile(A, X, theta)
        records.append({'orientation': 'horizontal', 'alpha': 0., 'theta_deg': theta,
                        **directed_barrier(state, positions, horizontal_id, theta, u)})
    profiles, barriers = pd.concat(curves, ignore_index=True), pd.DataFrame(records)
    np.savez_compressed(OUTPUT/'potential_profiles.npz',
                        columns=np.asarray(profiles.columns, dtype=str),
                        orientation=np.asarray(profiles.orientation, dtype=str),
                        values=profiles.drop(columns='orientation').to_numpy(dtype=float))
    barriers.to_csv(OUTPUT/'directed_barriers.csv', index=False)
    checks = verify(state, vertical_id)
    checks['parameter_quantities_match_source'] = verify_parameter_quantities()
    metadata = {'field_mT': FIELD_MT, 'normalization_temperature_K': REFERENCE_TEMPERATURE,
                'normalization_note': 'fixed energy scale; these landscapes do not depend on noise temperature',
                'radius_um': RADIUS, 'susceptibility': CHI, 'trap_sep_um': SEPARATION,
                'trap_height_pN_nm': HEIGHT*1000, 'trap_stiffness_pN_per_nm': STIFFNESS/1000,
                'C_pN_um4': COEFFICIENT, 'box_um': BOX, 'size': 10,
                'vertical_id': vertical_id, 'horizontal_id': horizontal_id,
                's_coordinate': 'positive points into initial occupied side; initial s=+d/2',
                'alpha': 'coherent interpolation: horizontal relative positions multiplied by (1-2*alpha); not a computed time trajectory',
                'geometry': 'all 199 other particles; nearest periodic image; no hard radial cutoff',
                'limitation': 'simulation has cutoff 200 um; a cutoff/image-convergence audit is required before rate predictions',
                'profile_offset': 'U(s)-U(+d/2), separately at each angle and stage',
                'barrier_definition': 'occupied positive-side minimum to intervening maximum separating opposite negative-side minimum',
                'checks': checks,
                'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [STATE, REPO/'src/parameters.py', Path(__file__), REFERENCE/'ModifiedPotential.ipynb', REFERENCE/'energy.py']}}
    (OUTPUT/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Saved profiles and directed barriers:', OUTPUT)
    print(barriers.query('orientation == "vertical" and theta_deg in [40,45,50,55,60]')
          [['alpha', 'theta_deg', 'status', 'barrier_kBTref']].to_string(index=False))
    return profiles, barriers


def verify_parameter_quantities():
    """Check the actual unit-bearing assignments without importing simulation wrappers."""
    import ast
    import types
    import pint
    source = (REPO/'src/parameters.py').read_text()
    assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)]
    registry = pint.UnitRegistry()
    namespace = {'np': np, 'ice': types.SimpleNamespace(ureg=registry)}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(REPO/'src/parameters.py'), 'exec'), namespace)
    p = namespace['params']
    for key, unit, expected in [('particle_radius', 'um', RADIUS), ('trap_sep', 'um', SEPARATION),
                                ('trap_height', 'pN*um', HEIGHT), ('trap_stiffness', 'pN/um', STIFFNESS),
                                ('particle_temperature', 'K', REFERENCE_TEMPERATURE)]:
        assert np.isclose(p[key].to(unit).magnitude, expected), key
    assert np.isclose(p['particle_susceptibility'], CHI)
    assert np.isclose((p['lattice_constant']*p['size']).to('um').magnitude, BOX)
    assert np.isclose(p['kb'].to('pN*um/K').magnitude, KB)
    return True


def verify(state, tag):
    positions = background(state, .5)
    a, x = geometry(state, positions, tag)
    # At theta=90 this equals the earlier -C*(3 rhat_x^2-1)/r^3 sum.
    s = 2.3
    j = int(np.flatnonzero(state.id.to_numpy() == tag)[0])
    axis = state.loc[j, ['dx', 'dy', 'dz']].to_numpy(dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    if axis@state.loc[j, ['cx', 'cy', 'cz']].to_numpy(dtype=float) < 0:
        axis *= -1
    point = state.loc[j, ['x', 'y', 'z']].to_numpy(dtype=float)+s*axis
    dr = point-positions
    dr -= BOX*np.rint(dr/BOX)
    r = np.linalg.norm(dr, axis=1)
    r[j] = np.inf
    legacy = np.sum(-COEFFICIENT/r**3*(3*(dr[:, 0]/r)**2-1))
    aa, xx = geometry(state, positions, tag, s=np.array([s]))
    assert np.isclose(legacy, COEFFICIENT*(aa[0]-3*xx[0]), rtol=1e-12)
    b = np.array([np.sin(np.deg2rad(47)), 0, np.cos(np.deg2rad(47))])
    e = dr/r[:, None]
    q = e@b
    force = np.sum(3*COEFFICIENT/r[:, None]**4*((1-5*q*q)[:, None]*e+2*q[:, None]*b), axis=0)@axis
    def energy(z):
        aa, xx = geometry(state, positions, tag, s=np.array([z]))
        return COEFFICIENT*(aa[0]-3*np.sin(np.deg2rad(47))**2*xx[0])
    numerical_force = -(energy(s+1e-5)-energy(s-1e-5))/(2e-5)
    assert np.isclose(force, numerical_force, rtol=1e-7)
    # Translation by a box vector must leave the energy unchanged.
    translated = state.copy()
    translated['x'] += BOX
    a2, x2 = geometry(translated, positions+np.array([BOX, 0, 0]), tag)
    assert np.allclose(a, a2) and np.allclose(x, x2)
    assert np.isclose(trap_potential(0), HEIGHT) and np.isclose(trap_potential(5), 0)
    return {'legacy_horizontal_field_identity': True, 'analytical_force_matches_energy_derivative': True,
            'periodic_translation_invariant': True, 'trap_height_and_minima': True}


def style():
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/frequencygs-matplotlib')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'text.usetex': True,
                         'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'lines.linewidth': 1.1})


def save(fig, stem):
    for extension in ['pdf', 'png']:
        fig.savefig(OUTPUT/f'{stem}.{extension}', dpi=300, bbox_inches='tight')


def plot_profiles(profiles, barriers):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    style()
    fig = plt.figure(figsize=(8.6/2.54, 6.1/2.54), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 25], wspace=.10)
    color_axis = fig.add_subplot(gs[0, :])
    norm = mpl.colors.Normalize(35, 60)
    cmap = plt.get_cmap('coolwarm')
    axes = []
    for column, alpha in enumerate([0., .5]):
        ax = fig.add_subplot(gs[1, column])
        axes.append(ax)
        for theta, g in profiles.query('alpha == @alpha').groupby('theta_deg'):
            ax.plot(g.s_um/SEPARATION, g.U_relative_kBTref, color=cmap(norm(theta)))
        ax.set_xlabel(r'$s/d$')
        ax.set_xticks([-.5, 0, .5], [r'$-1/2$', '$0$', r'$1/2$'])
        ax.set_xlim(-.565, .565)
        ax.set_title(rf'$\alpha={alpha:g}$', fontsize=10, pad=3)
        ax.text(.04, .96, ['a)', 'b)'][column], transform=ax.transAxes, va='top', fontsize=10)
        ax.axhline(0, color='.6', lw=.6, ls=':')
    axes[0].set_ylabel(r'$U_{\rm rel}/k_BT_{\rm ref}$', labelpad=0)
    # Same vertical scale in both panels, allowing the environment to be compared.
    inner = profiles.query('alpha in [0, .5] and abs(s_um) <= 5')
    low, high = inner.U_relative_kBTref.min(), inner.U_relative_kBTref.max()
    for ax in axes:
        ax.set_ylim(low-10, high+10)
    axes[1].tick_params(labelleft=False)
    inset = axes[0].inset_axes([.47, .62, .49, .25])
    g = barriers.query('orientation == "vertical" and alpha == 0 and theta_deg >= 35 and theta_deg <= 60')
    inset.plot(g.theta_deg, g.barrier_kBTref, color='darkslateblue', lw=.8)
    inset.set_xlabel(r'$\theta$', fontsize=7, labelpad=-7)
    inset.set_ylabel(r'$\tilde h$', fontsize=8, labelpad=-2)
    inset.set_xticks([35, 60])
    inset.tick_params(labelsize=6, pad=1)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, orientation='horizontal')
    cb.set_ticks([35, 40, 45, 50, 55, 60])
    color_axis.xaxis.set_ticks_position('top')
    color_axis.xaxis.set_label_position('top')
    cb.set_label(r'$\theta\,[\mathrm{deg}]$', labelpad=3)
    save(fig, 'rotating_vertical_potentials')
    return fig


def plot_barriers(barriers):
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6/2.54, 5.3/2.54), layout='constrained', sharex=True)
    for alpha, color, marker in zip(STAGES, ['darkslateblue', 'darkgoldenrod', '#aa3333'], ['D', 'v', 'o']):
        g = barriers.query('orientation == "vertical" and alpha == @alpha')
        axes[0].plot(g.theta_deg, g.barrier_kBTref, color=color, label=rf'$\alpha={alpha:g}$',
                     marker=marker, markevery=20, markersize=2)
        axes[1].plot(g.theta_deg, g.saddle_um/SEPARATION, color=color)
    axes[0].set_ylabel(r'$\Delta U/k_BT_{\rm ref}$', labelpad=0)
    axes[1].set_ylabel(r'$s_{\rm saddle}/d$', labelpad=0)
    axes[0].legend(frameon=False, fontsize=7, handlelength=1, loc='upper left', bbox_to_anchor=(0, .91))
    for ax, label in zip(axes, ['a)', 'b)']):
        ax.set_xlim(30, 90)
        ax.set_xlabel(r'$\theta\,[\mathrm{deg}]$')
        ax.set_xticks([30, 60, 90])
        ax.text(.04, .96, label, transform=ax.transAxes, va='top', fontsize=10)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(-.5, .5)
    save(fig, 'vertical_barriers_and_saddles')
    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    p, b = compute()
    plot_profiles(p, b)
    plot_barriers(b)
