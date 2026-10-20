"""Run one temperature-campaign PBS array element, with resource logging."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from run_local_campaign import run_one


def main():
    campaign = Path(os.environ['FREQUENCYGS_CAMPAIGN_ROOT']).resolve()
    settings = json.loads((campaign / 'campaign.json').read_text())
    if settings.get('pilot'):
        tau = settings['times'][0]
    else:
        tau = settings['times'][int(os.environ['PBS_ARRAY_INDEX'])]
    args = SimpleNamespace(**settings)
    record = run_one(tau, args, campaign)
    return record['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
