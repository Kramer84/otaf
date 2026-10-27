"""Compute the reference cases and save them as a named baseline.

Usage:
    python scripts/baselines/make_baseline.py --tag pre-overlap-fix
    python scripts/baselines/make_baseline.py --tag quick --only model1 model2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from baseline_cases import CASES, run, save  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="name of the baseline (file stem in data/)")
    parser.add_argument("--only", nargs="*", help="restrict to these case names")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.only or c.name in args.only]
    results = run(cases)
    print("saved", save(results, args.tag, cases))


if __name__ == "__main__":
    main()
