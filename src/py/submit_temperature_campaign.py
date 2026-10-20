"""Submit a short temperature pilot and a dependent PBS array."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def submit(command, directory):
    result = subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    print(f'Submitted {job_id}', flush=True)
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--times', type=float, nargs='+')
    parser.add_argument('--fields', type=int, nargs='+', default=list(range(21)))
    args = parser.parse_args()
    import math
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('temperature must be finite and positive')
    if args.times and any(not math.isfinite(t) or t <= 0 for t in args.times):
        parser.error('times must be finite and positive')
    if any(f < 0 for f in args.fields):
        parser.error('fields must be nonnegative')
    if Path(args.name).name != args.name or args.name in ('', '.', '..'):
        parser.error('name must be a single directory name')
    repo = Path(__file__).resolve().parents[2]
    # schedule long rotations first to reduce the final partially occupied wave
    times = sorted(set(args.times), reverse=True) if args.times else [100, 90, 80, 70, 50, 40, 30, *range(20, 0, -1)]
    settings = {'times': times, 'size': 10, 'realizations': 10,
                'temperature': args.temperature, 'framerate': 20, 'fields': sorted(set(args.fields)),
                'max_concurrent_jobs': 6, 'cpus_per_job': 10, 'memory_gib_per_job': 2}
    root = repo / 'data/campaigns' / args.name
    pilot = root.with_name(root.name + '-pilot')
    if root.exists() or pilot.exists():
        raise FileExistsError('Campaign or pilot already exists; choose a new name')
    subprocess.run([sys.executable, '-c',
                    'import numpy, pandas, scipy, numba, pint, matplotlib, pyarrow, psutil'], check=True)
    for directory, specification in [(root, settings),
        (pilot, {**settings, 'times': [.1], 'fields': [20], 'framerate': 100, 'pilot': True})]:
        directory.mkdir(parents=True)
        (directory / 'campaign.json').write_text(json.dumps(specification, indent=2) + '\n')
        environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                     check=True, capture_output=True, text=True).stdout
        (directory / 'environment.txt').write_text(environment)
    script = repo / 'jobs/temperature-array.pbs'
    record = {'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
              'pbs_script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
              'pilot_root': str(pilot), 'campaign_root': str(root)}
    label = f'fqgs-{args.temperature:g}K'
    pilot_command = ['/opt/pbs/bin/qsub', '-N', label + '-pilot',
                     '-l', 'walltime=00:10:00',
                     '-v', f'FREQUENCYGS_CAMPAIGN_ROOT={pilot},FREQUENCYGS_REPO_ROOT={repo}', str(script)]
    pilot_id = submit(pilot_command, pilot)
    record.update(pilot_job_id=pilot_id, pilot_command=pilot_command)
    (root / 'submission.json').write_text(json.dumps(record, indent=2) + '\n')
    array_command = ['/opt/pbs/bin/qsub', '-N', label, '-J', f'0-{len(times)-1}%6',
                     '-W', f'depend=afterok:{pilot_id}',
                     '-v', f'FREQUENCYGS_CAMPAIGN_ROOT={root},FREQUENCYGS_REPO_ROOT={repo}', str(script)]
    array_id = submit(array_command, root)
    record.update(array_job_id=array_id, array_command=array_command)
    (root / 'submission.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Campaign: {root}\nPilot: {pilot_id}\nArray: {array_id}', flush=True)


if __name__ == '__main__':
    main()
