"""Coherent horizontal lattice energy and deterministic overdamped trajectories.

All 100 horizontals share s_h; the 100 verticals stay in their initial wells.
Uses the existing nearest-image/no-cutoff potential convention for this first model.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/frequencygs-matplotlib')
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import root, brentq
import rotating_potential_analysis as p

OUT = p.REPO/'data/figures/collective-horizontal'
GRID = np.unique(np.r_[np.linspace(-5.1, 5.1, 3001), -5., 0., 5.])
TAUS = [4., 7., 8., 10., 20., 30., 40., 100.]
GAMMA = p.KB*300/.125  # pN s/um, fixed reference mobility


class CollectivePath:
    def __init__(self):
        self.state = pd.read_csv(p.STATE)
        self.centers = self.state[['x', 'y', 'z']].to_numpy()
        self.relative = self.state[['cx', 'cy', 'cz']].to_numpy()
        self.horizontal = self.state.dx.abs().to_numpy() > .5
        self.nh = int(self.horizontal.sum())
        assert self.nh == 100
        self.velocity = np.zeros_like(self.relative)
        self.velocity[self.horizontal] = self.relative[self.horizontal]/5
        self.base = self.centers+self.relative
        self.base[self.horizontal] = self.centers[self.horizontal]
        i, j = np.triu_indices(len(self.state), 1)
        keep = self.horizontal[i] | self.horizontal[j]
        self.i, self.j = i[keep], j[keep]
        self.r0 = self.base[self.i]-self.base[self.j]
        self.v = self.velocity[self.i]-self.velocity[self.j]
        basis = np.vstack([self.geometry(s) for s in GRID])
        self.A = CubicSpline(GRID, basis[:, 0])
        self.X = CubicSpline(GRID, basis[:, 1])

    def geometry(self, s):
        dr = self.r0+s*self.v
        dr -= p.BOX*np.rint(dr/p.BOX)
        r = np.linalg.norm(dr, axis=1)
        assert np.all(r > 0)
        return np.array([np.sum(r**-3), np.sum(dr[:, 0]**2*r**-5)])/self.nh

    def energy(self, s, theta):
        """Energy per horizontal particle; s-independent vertical pair energy omitted."""
        w = np.sin(np.deg2rad(theta))**2
        return p.trap_potential(s)+p.COEFFICIENT*(self.A(s)-3*w*self.X(s))

    def trap_gradient(self, s):
        return np.where(np.abs(s) <= 5, -8*p.HEIGHT*s/p.SEPARATION**2,
                        p.STIFFNESS*(s-5*np.sign(s)))

    def gradient(self, s, theta):
        w = np.sin(np.deg2rad(theta))**2
        return self.trap_gradient(s)+p.COEFFICIENT*(self.A(s, 1)-3*w*self.X(s, 1))

    def curvature(self, s, theta):
        trap = -8*p.HEIGHT/p.SEPARATION**2 if abs(s) < 5 else p.STIFFNESS
        w = np.sin(np.deg2rad(theta))**2
        return trap+p.COEFFICIENT*(self.A(s, 2)-3*w*self.X(s, 2))

    def direct_force(self, s, theta):
        dr = self.r0+s*self.v
        dr -= p.BOX*np.rint(dr/p.BOX)
        r = np.linalg.norm(dr, axis=1)
        e = dr/r[:, None]
        b = np.array([np.sin(np.deg2rad(theta)), 0, np.cos(np.deg2rad(theta))])
        q = e@b
        fij = 3*p.COEFFICIENT/r[:, None]**4*((1-5*q*q)[:, None]*e+2*q[:, None]*b)
        forces = np.zeros_like(self.centers)
        np.add.at(forces, self.i, fij)
        np.add.at(forces, self.j, -fij)
        magnetic = (forces*self.velocity).sum()/self.nh
        return magnetic-self.trap_gradient(s)


def dynamics(model, tau, tolerance=1e-8):
    # The energy here is U_total/n_h, so gamma ds/dt = -d(U_total/n_h)/ds.
    def rhs(t, y):
        return [-model.gradient(y[0], t*90/tau)/GAMMA]
    def cross(t, y):
        return y[0]
    cross.direction = -1
    cross.terminal = False
    sol = solve_ivp(rhs, [0, tau], [5.], method='Radau', rtol=tolerance,
                    atol=tolerance/10, max_step=tau/200, dense_output=True, events=cross)
    assert sol.success, sol.message
    times = np.arange(int(round(20*tau)))*.05
    s = sol.sol(times)[0]
    assert np.max(np.abs(s)) < 5.1
    events = sol.t_events[0]
    crossing_time = events[0] if len(events) else np.nan
    row = {'tau_s': tau, 'omega_rad_s': np.pi/(2*tau),
           'crossing_time_s': crossing_time, 'crossing_angle_deg': crossing_time*90/tau,
           's_final_um': s[-1], 'alpha_final': (5-s[-1])/10,
           'model': 'coherent deterministic horizontals; verticals fixed'}
    df = pd.DataFrame({'tau_s': tau, 'omega_rad_s': np.pi/(2*tau), 't_s': times,
                       'theta_deg': times*90/tau, 's_h_um': s, 'alpha': (5-s)/10,
                       'Rh_over_d': np.abs(s-5)/10})
    return df, row


def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    preserved = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in p.OUTPUT.glob('*') if f.suffix in ['.pdf', '.png']}
    p.verify_parameter_quantities()
    model = CollectivePath()
    profiles = []
    for theta in np.linspace(35, 60, 11):
        u = model.energy(GRID, theta)-model.energy(5., theta)
        profiles.append(pd.DataFrame({'theta_deg': theta, 's_h_um': GRID,
                                     'energy_relative_per_horizontal_pN_um': u,
                                     'energy_relative_per_horizontal_kBTref': u/(p.KB*300)}))
    profiles = pd.concat(profiles, ignore_index=True)
    np.savez_compressed(OUT/'collective_profiles.npz', columns=np.asarray(profiles.columns, dtype=str),
                        values=profiles.to_numpy())
    # Track the positive-side extrema. These are equilibrium configurations,
    # whereas the ODE below explicitly retains dynamical lag.
    equilibria = []
    for theta in np.linspace(0, 90, 361):
        energy = model.energy(GRID, theta)
        for i in find_peaks(-energy)[0]:
            equilibria.append({'theta_deg': theta, 's_h_um': GRID[i], 'type': 'minimum'})
        for i in find_peaks(energy)[0]:
            equilibria.append({'theta_deg': theta, 's_h_um': GRID[i], 'type': 'maximum'})
    equilibrium = pd.DataFrame(equilibria)
    equilibrium.to_csv(OUT/'equilibrium_extrema.csv', index=False)
    paths, summaries = [], []
    for tau in TAUS:
        path, summary = dynamics(model, tau)
        paths.append(path);summaries.append(summary)
    paths = pd.concat(paths, ignore_index=True)
    summary = pd.DataFrame(summaries)
    paths.to_csv(OUT/'horizontal_dynamics.csv', index=False)
    # Compare to existing cold horizontal crossing markers when available.
    observed = pd.read_csv(p.REPO/'data/analysis/left-boundary-timescales/timescales.csv')
    observed = observed.query('temperature_K == .1')[['total_time', 'horizontal_half_angle_deg']]
    summary = summary.merge(observed.rename(columns={'total_time': 'tau_s', 'horizontal_half_angle_deg': 'observed_cold_half_crossing_angle_deg'}), on='tau_s', how='left')
    summary['angle_difference_deg'] = summary.crossing_angle_deg-summary.observed_cold_half_crossing_angle_deg
    summary.to_csv(OUT/'switching_summary.csv', index=False)
    checks = verify(model)
    a, slow = dynamics(model, 8., tolerance=2e-10)
    reference = paths.query('tau_s == 8')
    discrepancy = float(np.max(np.abs(a.s_h_um.to_numpy()-reference.s_h_um.to_numpy())))
    assert discrepancy < 2e-5
    checks['tighter_integration_max_position_difference_um'] = discrepancy
    for name, digest in preserved.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    metadata = {'field_mT': 20, 'noise': 'none; deterministic reduced ODE', 'gamma_pN_s_per_um': GAMMA,
                'horizontal_particles': model.nh, 'vertical_particles': 100,
                'coordinate': 's_h=+5 um initially, 0 at hill, -5 um after flipping; alpha=(5-s_h)/10',
                'energy_normalization': 'total s_h-dependent lattice energy divided by 100; each pair counted once',
                'geometry': 'nearest periodic image; no hard radial cutoff; same approximation as earlier vertical scans',
                'background': 'vertical particles frozen at original well positions', 'checks': checks,
                'preserved_figures_sha256': preserved,
                'source_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                                  [Path(__file__), Path(p.__file__), p.STATE, p.REPO/'src/parameters.py']}}
    (OUT/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(summary.to_string(index=False))
    print('Checks:', checks)
    return profiles, paths, equilibrium, summary


def verify(model):
    for s, theta in [(2.3, 47), (-1.8, 55), (4.2, 30)]:
        direct = model.direct_force(s, theta)
        derivative = -model.gradient(s, theta)
        assert np.isclose(direct, derivative, rtol=2e-6, atol=1e-8), (direct, derivative)
        numerical = -(model.energy(s+1e-5, theta)-model.energy(s-1e-5, theta))/(2e-5)
        assert np.isclose(numerical, direct, rtol=2e-6, atol=1e-8)
    # Dividing energy by n_h must give the same rate as dividing total force
    # by n_h*gamma; direct_force above sums forces on all moving particles.
    assert np.isclose(GAMMA, .033135576)
    return {'projected_lattice_force_equals_energy_gradient': True,
            'finite_difference_energy_gradient': True, 'drag_and_collective_normalization': True,
            'source_parameter_quantities': True}


def save(fig, stem):
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT/f'{stem}.{suffix}', dpi=300, bbox_inches='tight')


def plot_potential(profiles, equilibrium):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    p.style()
    fig = plt.figure(figsize=(8.6/2.54, 6.1/2.54), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 25], wspace=.10)
    cbax = fig.add_subplot(gs[0, :])
    norm = mpl.colors.Normalize(35, 60)
    cmap = plt.get_cmap('coolwarm')
    left, right = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for theta, g in profiles.groupby('theta_deg'):
        left.plot(g.s_h_um/10, g.energy_relative_per_horizontal_kBTref, color=cmap(norm(theta)))
    inner = profiles.query('abs(s_h_um) <= 5').energy_relative_per_horizontal_kBTref
    left.set_ylim(inner.min()-5, inner.max()+5)
    left.set_xlim(-.51, .51)
    left.set_xticks([-.5, 0, .5], [r'$-1/2$', '$0$', r'$1/2$'])
    left.set_xlabel(r'$s_h/d$')
    left.set_ylabel(r'$U_{\rm rel}/(n_h k_BT_{\rm ref})$', labelpad=0)
    for typ, marker, color, label in [('minimum', '.', 'darkslateblue', 'minimum'),
                                      ('maximum', '.', 'darkgoldenrod', 'maximum')]:
        g = equilibrium.query('type == @typ')
        right.plot(g.theta_deg, g.s_h_um/10, marker, color=color, ms=2, label=label)
    right.set_xlim(30, 60);right.set_ylim(-.51, .51)
    right.set_xticks([30, 45, 60]);right.set_xlabel(r'$\theta\,[\mathrm{deg}]$')
    right.set_ylabel(r'$s_h/d$', labelpad=0)
    right.legend(frameon=False, fontsize=6, handlelength=.6, loc='center right')
    for ax, letter in [(left, 'a)'), (right, 'b)')]:
        ax.text(.04, .96, letter, transform=ax.transAxes, va='top', fontsize=10)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax, orientation='horizontal')
    cb.set_ticks([35, 40, 45, 50, 55, 60]);cb.set_label(r'$\theta\,[\mathrm{deg}]$', labelpad=3)
    cbax.xaxis.set_ticks_position('top');cbax.xaxis.set_label_position('top')
    save(fig, 'collective_horizontal_potential')
    return fig


def plot_dynamics(paths, summary):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    p.style()
    fig = plt.figure(figsize=(8.6/2.54, 6.1/2.54), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 25], wspace=.15)
    cbax = fig.add_subplot(gs[0, :])
    left, right = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    norm = mpl.colors.Normalize(0, .4);cmap = plt.get_cmap('RdYlGn')
    for tau, g in paths.groupby('tau_s'):
        left.plot(g.theta_deg, g.alpha, color=cmap(norm(np.pi/(2*tau))))
    left.axhline(.5, color='.6', ls='--', lw=.7)
    left.set_xlim(30, 70);left.set_ylim(-.02, 1.05)
    left.set_xticks([30, 50, 70]);left.set_xlabel(r'$\theta\,[\mathrm{deg}]$')
    left.set_ylabel(r'$\alpha(\theta)$', labelpad=0)
    right.plot(summary.omega_rad_s, summary.crossing_angle_deg, 'o-', color='darkslateblue', ms=2, label='model')
    right.plot(summary.omega_rad_s, summary.observed_cold_half_crossing_angle_deg, 'v', color='darkgoldenrod', ms=3, label='cold data')
    right.set_xlim(0, .4);right.set_xticks([0, .2, .4])
    right.set_xlabel(r'$\omega\,[\mathrm{rad/s}]$', fontsize=9)
    right.set_ylabel(r'$\theta_{\rm cross}\,[\mathrm{deg}]$', labelpad=0)
    right.legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0, .9), handlelength=.6)
    for ax, letter in [(left, 'a)'), (right, 'b)')]:
        ax.text(.04, .96, letter, transform=ax.transAxes, va='top', fontsize=10)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax, orientation='horizontal')
    cb.set_ticks([0, .2, .4]);cb.set_label(r'$\omega\,[\mathrm{rad/s}]$', labelpad=3)
    cbax.xaxis.set_ticks_position('top');cbax.xaxis.set_label_position('top')
    save(fig, 'collective_horizontal_dynamics')
    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    profiles, paths, equilibrium, summary = compute()
    plot_potential(profiles, equilibrium)
    plot_dynamics(paths, summary)
