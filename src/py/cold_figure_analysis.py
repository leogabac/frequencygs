"""Saved-data reductions for the cold paper-figure notebook."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def pack_paths(data_root, size=10):
    return {float(p.name): p for p in Path(data_root).glob(f'*/test11_improved/{size}/*')
            if p.is_dir()}


def motion_series(data_root, output_root, field=20, size=10, trap_sep=10):
    """Cache realization-resolved instantaneous crossings and initial displacements."""
    packs = pack_paths(data_root, size)
    sources = [(tau, realization, pack / f'{field}mT/xtrj{realization}.parquet')
               for tau, pack in sorted(packs.items()) for realization in range(1, 11)]
    signature = {'algorithm': 'initial-projected-displacement-v1', 'field': field,
                 'size': size, 'trap_sep_um': trap_sep,
                 'sources': [{'path': str(p), 'bytes': p.stat().st_size,
                              'mtime_ns': p.stat().st_mtime_ns} for _, _, p in sources]}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    output_root = Path(output_root)
    cache = output_root / f'motion_{field}mT.parquet'
    manifest = cache.with_suffix('.json')
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text())['signature'] == digest:
        return pq.read_table(cache).to_pandas()
    records = []
    for tau, realization, path in sources:
        df = pq.read_table(path, columns=['frame', 'id', 't', 'dx', 'dy', 'dz', 'cx', 'cy', 'cz']).to_pandas()
        df = df.sort_values(['frame', 'id'])
        count = 2 * size**2
        assert len(df) == int(round(20*tau))*count
        assert np.array_equal(df.id.to_numpy().reshape(-1, count),
                              np.tile(np.arange(count), (len(df)//count, 1)))
        dirs = df[['dx', 'dy', 'dz']].to_numpy().reshape(-1, count, 3)
        axis = np.abs(dirs / np.linalg.norm(dirs, axis=2, keepdims=True))
        assert np.allclose(axis, axis[0])
        horizontal = axis[0, :, 0] > 0.5
        assert horizontal.sum() == size**2
        rels = df[['cx', 'cy', 'cz']].to_numpy().reshape(-1, count, 3)
        parallel = np.sum(rels * axis, axis=2)
        # Same criterion as take_distances.py + figure4.ipynb: displacement >= d/2.
        # This counts the particles currently across the hill, not all past events.
        distance = np.abs(parallel - parallel[0])
        assert np.allclose(np.abs(parallel[0]), trap_sep/2)
        times = df.t.to_numpy().reshape(-1, count)
        assert np.allclose(times, times[:, :1])
        records.append(pd.DataFrame({'total_time': tau, 'realization': realization,
            't': times[:, 0], 'nu': (distance >= trap_sep/2).mean(axis=1),
            'Rh': distance[:, horizontal].mean(axis=1)/trap_sep,
            'Rv': distance[:, ~horizontal].mean(axis=1)/trap_sep}))
        if realization == 10:
            print(f'Motion: tau={tau:g} s, 10 realizations reduced', flush=True)
    result = pd.concat(records, ignore_index=True)
    temporary = cache.with_suffix('.parquet.tmp')
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), temporary, compression='zstd')
    temporary.replace(cache)
    manifest.write_text(json.dumps({'signature': digest, **signature}, indent=2)+'\n')
    return result


def snapshot(data_root, tau, field=20, realization=1, size=10):
    pack = pack_paths(data_root, size)[float(tau)] / f'{field}mT'
    trajectory = pq.read_table(pack / f'xtrj{realization}.parquet').to_pandas()
    vertex_file = pq.ParquetFile(pack / f'vertices{realization}.parquet')
    vertices = vertex_file.read_row_group(vertex_file.num_row_groups-1).to_pandas()
    last = trajectory.frame.max()
    vertices = vertices.loc[vertices.frame == last].copy()
    assert len(vertices) == size**2
    assert np.allclose(vertices.t, tau-0.05)
    return trajectory, vertices


def activity_widths(averaged, smooth_window=5, tail_fraction=0.08):
    """Full active span, using the exact envelope definition in figure4.ipynb."""
    rows = []
    for tau, group in averaged.groupby('total_time'):
        group = group.sort_values('t')
        theta = group.t.to_numpy()*90/tau
        signal = np.diff(group.nu.to_numpy())
        envelope = np.convolve(np.abs(signal), np.ones(smooth_window)/smooth_window, mode='same')
        active = np.flatnonzero(envelope >= tail_fraction*envelope.max()) if envelope.max() > 0 else []
        mid = (theta[1:]+theta[:-1])/2
        width = mid[active[-1]]-mid[active[0]] if len(active) else np.nan
        rows.append({'total_time': tau, 'omega': np.pi/(2*tau), 'width_deg': width})
    return pd.DataFrame(rows).sort_values('omega')
