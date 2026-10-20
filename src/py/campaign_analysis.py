"""Postprocessing for one rotation-time pack; raw trajectories stay untouched."""

from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
import vertices as vrt
from trajectory_storage import read_table, write_parquet_frames


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.csv.tmp')
    os.close(fd)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def clean_trajectory(field_path, realization, storage_format="csv"):
    field_path = Path(field_path)
    source = field_path / 'trj' / f'trj{realization}'
    if not any(source.with_suffix('.' + ext).exists() for ext in ('parquet', 'csv')):
        source = field_path / f'trj{realization}'
    df = read_table(source, storage_format).set_index(['frame', 'id'])
    if 'mux' in df:
        # retain the original campaign's last-frame removal
        last = df.index.get_level_values('frame').unique()[-1]
        df = df.loc[df.index.get_level_values('frame') != last]
    df = df.drop(columns=['mux', 'muy', 'muz', 'type'], errors='ignore')
    if df.empty or df.index.has_duplicates:
        raise ValueError(f'Empty trajectory or duplicate particle rows: {source}')
    frames = df.index.get_level_values('frame').unique()
    ids = df.index.get_level_values('id').unique()
    for frame in frames:
        if set(df.xs(frame).index) != set(ids):
            raise ValueError(f'Incomplete frame {frame}: {source}')
    # map labels rather than assuming the rows form an ordered Cartesian product
    df = df.reset_index()
    df['frame'] = df['frame'].map(dict(zip(frames, range(len(frames)))))
    df['id'] = df['id'].map(dict(zip(ids, range(len(ids)))))
    df = df.sort_values(['frame', 'id'])
    if storage_format == 'parquet':
        write_parquet_frames([df], field_path / f'xtrj{realization}.parquet')
    else:
        save_csv(df, field_path / f'xtrj{realization}.csv')


def compute_vertices(field_path, realization, size, lattice_constant, storage_format="csv"):
    field_path = Path(field_path)
    df = read_table(field_path / f'xtrj{realization}', storage_format)
    lattice = vrt.create_lattice(lattice_constant, size)
    coords = lattice.reshape(-1, 3)
    def frames():
        for frame, rows in df.groupby('frame', sort=True):
            centers, dirs, rels = vrt.trj2numpy(rows)
            norms = np.linalg.norm(dirs, axis=1)
            if np.any(norms == 0) or not np.isfinite(norms).all():
                raise ValueError(f'Invalid ice directions at frame {frame}')
            dirs = dirs / norms[:, None]
            indices = vrt.indices_lattice(lattice, centers, lattice_constant, size)
            charges = vrt.get_charge_lattice(indices, dirs)
            dipoles = vrt.dipole_lattice(centers, dirs, rels, lattice, indices)
            dipoles *= (charges == 0)[:, :, None]
            times = rows['t'].unique()
            if len(times) != 1:
                raise ValueError(f'Inconsistent timestamps at frame {frame}')
            yield pd.DataFrame({
                'frame': frame, 'vertex': np.arange(size * size),
                'x': coords[:, 0], 'y': coords[:, 1], 'coordination': 4,
                'charge': charges.ravel(), 'dx': dipoles.reshape(-1, 3)[:, 0],
                'dy': dipoles.reshape(-1, 3)[:, 1], 't': times[0],
            })

    if storage_format == 'parquet':
        write_parquet_frames(frames(), field_path / f'vertices{realization}.parquet')
        return
    output = field_path / f'vertices{realization}.csv'
    fd, temporary = tempfile.mkstemp(dir=field_path, suffix='.vertices.tmp')
    os.close(fd)
    try:
        first = True
        for result in frames():
            result.to_csv(temporary, mode='a', header=first, index=False)
            first = False
        if first:
            raise ValueError(f'Empty cleaned trajectory: {field_path}')
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def analyze_vertices(field_paths, realizations, size, total_time, *, kappa, averages, storage_format="csv"):
    kappa_rows, fraction_rows = [], []
    signs = (-1) ** np.indices((size, size)).sum(axis=0)
    for field_path in field_paths:
        field_path = Path(field_path)
        field = float(field_path.name[:-2])
        for realization in realizations:
            vertices = read_table(field_path / f'vertices{realization}', storage_format)
            for frame, rows in vertices.groupby('frame', sort=True):
                rows = rows.sort_values('vertex')
                if not np.array_equal(rows.vertex.to_numpy(), np.arange(size * size)):
                    raise ValueError(f'Incomplete vertices: {field_path}, frame {frame}')
                t = rows.t.iloc[0]
                q = rows.charge.to_numpy()
                if kappa:
                    kappa_rows.append([realization, t, np.sum(q * signs.ravel()), field, total_time])
                if averages:
                    dipole = rows.dx.to_numpy() ** 2 + rows.dy.to_numpy() ** 2
                    fractions = [np.mean(q == -4), np.mean(q == -2),
                                 np.mean((q == 0) & (dipole == 0)),
                                 np.mean((q == 0) & (dipole > 0)),
                                 np.mean(q == 2), np.mean(q == 4)]
                    fraction_rows.append([*fractions, t, field, total_time])
    return (pd.DataFrame(kappa_rows, columns=['realization', 't', 'kappa', 'field', 'total_time']),
            pd.DataFrame(fraction_rows, columns=['I', 'II', 'III', 'IV', 'V', 'VI',
                                                'time', 'field', 'total_time'])
            .groupby(['time', 'field', 'total_time'], as_index=False).mean())
