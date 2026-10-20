"""Compare saved warm/cold motion at 20 mT; does not launch simulations."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/frequencygs-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
WARM = REPO / 'data/test11/10'
COLD = REPO / 'data/figures/cold-0p1K'
OUT = REPO / 'data/analysis/left-boundary-timescales'


def envelope_span(t, values, window, fraction):
    signal = np.diff(values)
    envelope = np.convolve(np.abs(signal), np.ones(window)/window, mode='same')
    active = np.flatnonzero(envelope >= fraction * envelope.max()) if envelope.max() else []
    mid = (t[1:] + t[:-1])/2
    return (mid[active[0]], mid[active[-1]]) if len(active) else (np.nan, np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(WARM / 'countshorrp_20mT.csv')
    v = pd.read_csv(WARM / 'countsverrp_20mT.csv')
    assert h[['total_time', 't']].equals(v[['total_time', 't']])
    assert h.groupby(['total_time', 't']).size().eq(10).all()
    motion = h[['total_time', 't']].copy()
    motion['nu'] = (h.counts + v.counts)/200
    motion['fh'] = h.counts/100
    motion['fv'] = v.counts/100
    motion = motion.groupby(['total_time', 't'], as_index=False).mean()
    for axis, name in [('Rh', 'horavrp.csv'), ('Rv', 'veravrp.csv')]:
        d = pd.read_csv(WARM/name)
        assert d.field.eq(20).all()
        d = d.groupby(['total_time', 't'], as_index=False)['mean'].mean()
        motion = motion.merge(d.rename(columns={'mean': axis}), validate='one_to_one')
        motion[axis] /= 10
    k = pd.read_csv(WARM/'kappa.csv')
    assert k.field.eq(20).all()
    k = k.loc[k.groupby(['total_time', 'realization']).t.idxmax()].copy()
    k['kappa'] = k.kappa.abs()/400
    endpoints = k.groupby('total_time', as_index=False).agg(kappa=('kappa', 'mean'), kappa_std=('kappa', 'std'))
    warm = motion.merge(endpoints, on='total_time', validate='many_to_one')
    warm['temperature_K'] = 300.

    cold = pd.read_parquet(COLD/'motion_20mT.parquet')
    cold = cold.groupby(['total_time', 't'], as_index=False)[['nu', 'Rh', 'Rv']].mean()
    # The saved motion cache does not separate crossing fractions by orientation.
    # Read those columns from the same 270 source trajectories, without altering them.
    cache = OUT/'cold_orientation_crossings.csv'
    manifest = json.loads((COLD/'motion_20mT.json').read_text())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache_meta = OUT/'cold_orientation_crossings.json'
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())['signature'] == signature:
        orient = pd.read_csv(cache, float_precision='round_trip')
    else:
        records = []
        for item in manifest['sources']:
            p = Path(item['path'])
            assert p.stat().st_size == item['bytes'] and p.stat().st_mtime_ns == item['mtime_ns']
            df = pq.read_table(p, columns=['frame', 'id', 't', 'dx', 'dy', 'dz', 'cx', 'cy', 'cz']).to_pandas().sort_values(['frame', 'id'])
            assert len(df) % 200 == 0
            axes = np.abs(df[['dx', 'dy', 'dz']].to_numpy().reshape(-1, 200, 3))
            axes /= np.linalg.norm(axes, axis=2, keepdims=True)
            assert np.allclose(axes, axes[0])
            horizontal = axes[0, :, 0] > .5
            assert horizontal.sum() == 100
            rel = df[['cx', 'cy', 'cz']].to_numpy().reshape(-1, 200, 3)
            parallel = (rel*axes).sum(2)
            crossed = np.abs(parallel-parallel[0]) >= 5
            times = df.t.to_numpy().reshape(-1, 200)[:, 0]
            records.append(pd.DataFrame({'total_time': float(p.parent.parent.name), 't': times,
                                         'fh': crossed[:, horizontal].mean(1),
                                         'fv': crossed[:, ~horizontal].mean(1)}))
        orient = pd.concat(records).groupby(['total_time', 't'], as_index=False).mean()
        orient.to_csv(cache, index=False)
        cache_meta.write_text(json.dumps({'signature': signature, 'sources': manifest['sources']}, indent=2)+'\n')
    retained_rows = len(cold)
    cold = cold.merge(orient, validate='one_to_one')
    assert len(cold) == retained_rows
    assert np.allclose(cold.nu, (cold.fh+cold.fv)/2)
    endpoint = pd.read_csv(COLD/'final_observables.csv').query('field == 20')
    cold = cold.merge(endpoint[['total_time', 'kappa', 'kappa_std']], on='total_time', validate='many_to_one')
    cold['temperature_K'] = .1
    all_motion = pd.concat([warm, cold], ignore_index=True)
    all_motion.to_csv(OUT/'averaged_motion.csv', index=False)

    rows, sensitivities = [], []
    for (temperature, tau), g in all_motion.groupby(['temperature_K', 'total_time']):
        g = g.sort_values('t')
        t = g.t.to_numpy()
        lo, hi = envelope_span(t, g.nu.to_numpy(), 5, .08)
        # Compare the inset width to less tail-sensitive cumulative activity quantiles.
        activity = np.abs(np.diff(g.nu.to_numpy()))
        cumulative = np.cumsum(activity)/activity.sum()
        mid = (t[1:]+t[:-1])/2
        quantile = lambda q: mid[min(np.searchsorted(cumulative, q), len(mid)-1)]
        ih = np.flatnonzero(g.fh.to_numpy() >= .5)
        iv = np.flatnonzero(g.fv.to_numpy() >= .05)
        row = {'temperature_K': temperature, 'total_time': tau, 'omega': np.pi/(2*tau),
               'kappa': g.kappa.iloc[-1], 'kappa_std': g.kappa_std.iloc[-1],
               'fh_final': g.fh.iloc[-1], 'fv_final': g.fv.iloc[-1], 'fv_max': g.fv.max(),
               'activity_start_deg': lo*90/tau, 'activity_end_deg': hi*90/tau,
               'width_deg': (hi-lo)*90/tau, 'activity_duration_s': hi-lo,
               'activity_5_95_duration_s': quantile(.95)-quantile(.05),
               'horizontal_half_angle_deg': t[ih[0]]*90/tau if len(ih) else np.nan,
               'vertical_5pct_angle_deg': t[iv[0]]*90/tau if len(iv) else np.nan,
               'vertical_max_displacement_d': g.Rv.max(), 'vertical_final_displacement_d': g.Rv.iloc[-1]}
        rows.append(row)
        for window in [1, 3, 5, 9]:
            for fraction in [.04, .08, .16]:
                a, b = envelope_span(t, g.nu.to_numpy(), window, fraction)
                sensitivities.append({'temperature_K': temperature, 'total_time': tau,
                                      'window_frames': window, 'tail_fraction': fraction,
                                      'duration_s': b-a, 'width_deg': (b-a)*90/tau})
    summary = pd.DataFrame(rows).sort_values(['temperature_K', 'omega'])
    summary.to_csv(OUT/'timescales.csv', index=False)
    pd.DataFrame(sensitivities).to_csv(OUT/'width_sensitivity.csv', index=False)
    reference = pd.read_csv(COLD/'activity_widths.csv')
    compare = summary.query('temperature_K == .1').merge(reference, on='total_time', suffixes=('', '_reference'))
    assert np.allclose(compare.width_deg, compare.width_deg_reference)

    boundaries = []
    for temperature, group in summary.groupby('temperature_K'):
        group = group.sort_values('omega')
        for target in [.95, .99]:
            passes = np.flatnonzero(group.kappa.to_numpy() >= target)
            idx = passes[0]
            low, high = group.iloc[idx-1], group.iloc[idx]
            fraction = (target-low.kappa)/(high.kappa-low.kappa)
            boundaries.append({'temperature_K': temperature, 'kappa_target': target,
                               'omega_low': low.omega, 'omega_high': high.omega,
                               'omega_linear_interpolation': low.omega+fraction*(high.omega-low.omega)})
    pd.DataFrame(boundaries).to_csv(OUT/'boundary_brackets.csv', index=False)
    frozen_forces()
    plot(summary)
    provenance = {'field_mT': 20, 'warm_temperature_K': 300,
                  'warm_temperature_basis': 'paper and parameters.py; historical pack manifests unavailable',
                  'noise_temperature_cold_K': .1, 'gamma_pN_s_per_um': .033135576,
                  'saved_frame_spacing_s': .05, 'inset_window_frames': 5, 'inset_tail_fraction': .08,
                  'warm_angle': 'exact pi/(2*tau), rather than notebook omega rounded to 4 decimals',
                  'cold_motion_signature': signature,
                  'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [WARM/'countshorrp_20mT.csv', WARM/'countsverrp_20mT.csv', WARM/'kappa.csv',
                              WARM/'horavrp.csv', WARM/'veravrp.csv', COLD/'motion_20mT.parquet', COLD/'final_observables.csv']}}
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(summary.query('total_time in [7, 8, 9, 10, 30, 40, 100]')
          [['temperature_K', 'total_time', 'omega', 'kappa', 'fv_final', 'width_deg', 'activity_duration_s', 'activity_5_95_duration_s']].to_string(index=False))
    print(pd.DataFrame(boundaries).to_string(index=False))
    print('Outputs:', OUT)


def frozen_forces():
    """Diagnostic single-particle crossing, nearest periodic image, frozen ice neighbours."""
    state = pd.read_csv(REPO/'data/states/ice/10.csv')
    centers = state[['x', 'y', 'z']].to_numpy()
    relative = state[['cx', 'cy', 'cz']].to_numpy()
    axes = np.abs(state[['dx', 'dy', 'dz']].to_numpy())
    j = np.flatnonzero(axes[:, 1] > .5)[0]
    sign = np.sign(relative[j, 1])
    positions = centers+relative
    C = 23040.  # pN um^4, from radius 5 um, chi .0576, B .020 T
    rows = []
    for theta in [30, 35.26, 40, 42, 45, 50]:
        b = np.array([np.sin(np.deg2rad(theta)), 0, np.cos(np.deg2rad(theta))])
        for distance in np.linspace(0, 5, 101):
            pos = positions[j]-sign*distance*axes[j]
            dr = pos-positions
            dr -= 300*np.rint(dr/300)
            radius = np.linalg.norm(dr, axis=1)
            radius[j] = np.inf
            e = dr/radius[:, None]
            q = e@b
            f = (3*C/radius[:, None]**4*((1-5*q*q)[:, None]*e+2*q[:, None]*b))
            f[radius > 200] = 0
            dipolar = -sign*f.sum(0)[1]
            trap = -.00032*(5-distance)
            rows.append({'theta_deg': theta, 'displacement_um': distance,
                         'dipolar_force_toward_hill_pN': dipolar,
                         'trap_force_toward_hill_pN': trap,
                         'total_force_toward_hill_pN': dipolar+trap})
    pd.DataFrame(rows).to_csv(OUT/'frozen_vertical_force.csv', index=False)


def plot(summary):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for temperature, color in [(300, '#bb3322'), (.1, '#2266aa')]:
        g = summary.query('temperature_K == @temperature').sort_values('omega')
        label = f'{temperature:g} K'
        axes[0, 0].plot(g.omega, g.kappa, 'o-', ms=3, color=color, label=label)
        axes[0, 1].plot(g.omega, g.fv_final, 'o-', ms=3, color=color)
        axes[1, 0].plot(g.omega, g.activity_duration_s, 'o-', ms=3, color=color, label=label+' inset width')
        durations = g.activity_5_95_duration_s.where(g.activity_5_95_duration_s > 0)
        axes[1, 0].plot(g.omega, durations, '--', color=color, label=label+' 5–95% activity')
        axes[1, 1].plot(g.fv_final, 1-g.kappa, 'o', ms=4, color=color)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlim(.001, .4)
        ax.set_xlabel(r'$\omega$ [rad/s]')
    axes[0, 0].set_ylabel(r'Final $\langle|\kappa|\rangle$')
    axes[0, 0].legend()
    axes[0, 1].set_ylabel('Final vertical crossing fraction')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel('Activity duration [s]')
    axes[1, 0].axhline(.2, color='.5', ls=':', label='Five-frame single-step span')
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot([0, .5], [0, .5], color='.5', ls='--')
    axes[1, 1].set_xlabel('Final vertical crossing fraction')
    axes[1, 1].set_ylabel(r'$1-\langle|\kappa|\rangle$')
    fig.suptitle('20 mT: slow-driving defects and measured activity times')
    fig.savefig(OUT/'timescale_comparison.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
