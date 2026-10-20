"""Compressed trajectory tables, with bounded-memory and atomic writes."""

from pathlib import Path
import os
import tempfile

import pandas as pd


def require_parquet():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError('Parquet storage requires pyarrow in the simulation environment') from exc
    return pa, pq


def write_parquet_frames(frames, path, batch_rows=65536):
    """Write incoming frames in Zstandard-compressed row groups, without CSV staging."""
    pa, pq = require_parquet()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.parquet.tmp')
    os.close(fd)
    writer = None
    pending, rows = [], 0

    def flush():
        nonlocal writer, pending, rows
        df = pd.concat(pending, ignore_index=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
        writer.write_table(table, row_group_size=batch_rows)
        pending, rows = [], 0

    try:
        for frame in frames:
            if frame.empty:
                continue
            if isinstance(frame.index, pd.MultiIndex):
                frame = frame.reset_index()
            pending.append(frame)
            rows += len(frame)
            if rows >= batch_rows:
                flush()
        if pending:
            flush()
        if writer is None:
            raise ValueError(f'No rows to write: {path}')
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_table(stem, preferred='parquet'):
    """Prefer the requested format, falling back to a historical CSV table."""
    stem = Path(stem)
    formats = [preferred, 'csv' if preferred == 'parquet' else 'parquet']
    for extension in formats:
        path = stem.with_suffix('.' + extension)
        if path.exists():
            if extension == 'parquet':
                _, pq = require_parquet()
                df = pq.read_table(path).to_pandas()
            else:
                df = pd.read_csv(path)
            return df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    raise FileNotFoundError(f'No trajectory table at {stem}.parquet or {stem}.csv')


def convert_lammps_trajectory(col, path):
    import icenumerics as ice

    col.sim.load(read_trj=False)
    frames = (ice.get_ice_trj_single(col, i)[0]
              for i in range(len(col.sim.lazy_read.T)))
    write_parquet_frames(frames, path)
