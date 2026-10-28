"""Recompute the verification cases and regenerate the tables of docs/guide/verification.md.

Usage:
    python scripts/verification/run_verification.py                # everything (~10 min, scipy only)
    python scripts/verification/run_verification.py --only dumas   # one group, the others are kept
    python scripts/verification/run_verification.py --render-only  # tables from the saved JSON
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import verification_cases as vc  # noqa: E402


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=["dumas", "thesis", "baselines"], help="groups to recompute")
    parser.add_argument("--render-only", action="store_true", help="only rewrite the tables from the saved results")
    args = parser.parse_args()

    if not args.render_only:
        groups = args.only or ["dumas", "thesis", "baselines"]
        fresh = vc.run(groups)
        results = vc.load() if vc.RESULTS_PATH.exists() and args.only else {}
        provenance = {**results.get("provenance", {}), **fresh.pop("provenance")}
        results.update(fresh, provenance=provenance)
        vc.save(results)
        print("saved", vc.RESULTS_PATH)
    vc.TABLES_PATH.write_text(vc.render_tables(vc.load()))
    print("wrote", vc.TABLES_PATH)


if __name__ == "__main__":
    main()
