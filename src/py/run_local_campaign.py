"""Run independent rotation times with a fixed concurrency and resource logging."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def run_one(tau, args, campaign):
    label = f'tau-{tau:g}'
    directory = campaign / 'timings' / label
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(Path(__file__).with_name('test11_improved.py')),
               str(args.size), str(tau), '--temperature', str(args.temperature),
               '--realizations', str(args.realizations), '--framerate', str(args.framerate),
               '--fields', *map(str, args.fields), '--data-root', str(campaign),
               '--lammps-root', str(campaign / 'lammps')]
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[name] = '1'
    env['MPLBACKEND'] = 'Agg'
    start = time.monotonic()
    record = {'tau_seconds': tau, 'command': command,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'peak_process_tree_rss_bytes': 0, 'peak_processes': 0,
              'rss_note': 'sampled sum of parent and descendant RSS; shared pages can be counted twice'}
    with (directory / 'stdout.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        record['pid'] = process.pid
        root = psutil.Process(process.pid)
        while process.poll() is None:
            rss, count = 0, 0
            try:
                children = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                children = []
            for child in children:
                try:
                    rss += child.memory_info().rss
                    count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            record['peak_process_tree_rss_bytes'] = max(record['peak_process_tree_rss_bytes'], rss)
            record['peak_processes'] = max(record['peak_processes'], count)
            time.sleep(0.2)
        record['returncode'] = process.returncode
    record['wall_seconds'] = time.monotonic() - start
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (directory / 'resources.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'{label}: exit={record["returncode"]}, wall={record["wall_seconds"]:.1f}s, '
          f'peak tree RSS={record["peak_process_tree_rss_bytes"] / 2**30:.2f} GiB', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--times', type=float, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--realizations', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--framerate', type=float, default=20)
    parser.add_argument('--fields', type=float, nargs='+', default=list(range(21)))
    parser.add_argument('--data-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'data' / 'campaigns')
    parser.add_argument('--name', default='cold-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    if args.jobs < 1 or len(set(args.times)) != len(args.times):
        parser.error('jobs must be positive and times must be unique')
    if any(not math.isfinite(tau) or tau <= 0 for tau in args.times):
        parser.error('rotation times must be finite and positive')
    if Path(args.name).name != args.name or args.name in ('', '.', '..'):
        parser.error('name must be a single directory name')
    campaign = args.data_root.resolve() / args.name
    campaign.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    (campaign / 'campaign.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(f'Campaign: {campaign}', flush=True)
    print(f'Up to {args.jobs} jobs, {args.realizations} LAMMPS processes each', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, tau, args, campaign) for tau in args.times]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record['tau_seconds'])
    (campaign / 'resources.json').write_text(json.dumps(records, indent=2) + '\n')
    return int(any(record['returncode'] for record in records))


if __name__ == '__main__':
    raise SystemExit(main())
