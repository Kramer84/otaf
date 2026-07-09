#!/usr/bin/env python3
"""
pydeps_command.py — robust wrapper around pydeps for the OTAF package.

- Calls pydeps on the *directory* (e.g. ./src/otaf) so module discovery works.
- Hides noisy externals you listed by default.
- Optional focus on a subpackage via --subpath (uses --only otaf.<subpath>).
- Private filtering (--exclude-private) auto-disables when focusing a private
  subpackage like `_assembly_modeling`.
- `--include-missing` can dodge rare KeyError cycles by keeping placeholder nodes.

Examples
--------
# Full package, safe defaults
python pydeps_command.py \
  --src ./src --package otaf \
  --out build/deps/otaf_full.svg \
  --depth 2 --rankdir LR

# Focused view
python pydeps_command.py \
  --src ./src --package otaf --subpath _assembly_modeling \
  --out build/deps/assembly.svg \
  --depth 2 --rankdir LR
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


DEFAULT_EXCLUDES = [
    # User-requested externals
    "beartype*",
    "numpy*",
    "matplotlib*",
    "IPython*",
    "sklearn*",
    "torcheval*",
    "tqdm*",
    "pytransform3d*",
    "torch*",
    "trimesh*",
    # Common hubs that add clutter
    "typing*",
    "collections*",
]


def build_excludes(
    package: str,
    subpath: str | None,
    exclude_private: bool,
    extra_excludes: Sequence[str],
) -> List[str]:
    """Assemble -x patterns, being careful with private targets.

    If focusing a private subpackage (name starts with `_`), don't exclude it.
    """
    patterns: List[str] = list(DEFAULT_EXCLUDES)
    patterns.extend(extra_excludes)

    is_private_target = bool(subpath and subpath.lstrip(".").startswith("_"))
    if exclude_private and not is_private_target:
        # Hide private top-level subpackages and any deeper private segments
        patterns.extend([f"{package}._*", f"{package}.*._*"])
    return patterns


def run_pydeps(
    src_root: Path,
    package: str,
    out: Path,
    subpath: str | None,
    depth: int,
    rankdir: str,
    exclude_private: bool,
    extra_excludes: Sequence[str],
    noise_level: int,
    include_missing: bool = False,
    cluster: bool = True,
    use_only: bool = True,
) -> None:
    """Invoke pydeps with robust defaults for large projects.

    Parameters
    ----------
    src_root : Path
        Directory that contains the package (e.g., ./src).
    package : str
        Top-level package name (e.g., otaf).
    out : Path
        Output SVG path.
    subpath : str | None
        Optional subpackage to focus (e.g., _assembly_modeling).
    depth : int
        Value for --max-module-depth.
    rankdir : str
        TB | BT | LR | RL.
    exclude_private : bool
        Exclude private segments (._*) unless focusing a private target.
    extra_excludes : Sequence[str]
        Additional -x patterns.
    noise_level : int
        Degree threshold for pruning; 0 disables.
    include_missing : bool
        Include modules not found on sys.path (workaround for rare cycles).
    cluster : bool
        Draw external deps as clusters.
    use_only : bool
        Pass --only to scope the graph to the package/subpackage.
    """
    target_dir = (src_root / package).resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {target_dir}")

    only_arg = package if not subpath else f"{package}.{subpath.lstrip('.')}"
    excludes = build_excludes(package, subpath, exclude_private, extra_excludes)

    out.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "pydeps",
        str(target_dir),
        "--noshow",
        "--rankdir",
        rankdir,
        "--max-bacon",
        "2",
        "--max-module-depth",
        str(depth),
        "--rmprefix",
        f"{package}.",
        "-o",
        str(out),
    ]

    if use_only:
        cmd.extend(["--only", only_arg])

    if cluster:
        cmd.extend(["--cluster", "--max-cluster-size", "1000", "--min-cluster-size", "2"])

    if excludes:
        cmd.extend(["-x", *excludes])

    if noise_level and noise_level > 0:
        cmd.extend(["--noise-level", str(noise_level)])

    if include_missing:
        cmd.append("--include-missing")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_root.resolve())] + ([env["PYTHONPATH"]] if "PYTHONPATH" in env else [])
    )

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # Show both streams and exit with the same code
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate readable pydeps graphs for the OTAF package.")
    p.add_argument("--src", type=Path, required=True, help="Root that contains the package (e.g., ./src)")
    p.add_argument("--package", default="otaf", help="Top-level package name (default: otaf)")
    p.add_argument("--subpath", default=None, help="Optional subpackage focus (e.g., _assembly_modeling)")
    p.add_argument("--out", type=Path, required=True, help="Output SVG file path")
    p.add_argument("--depth", type=int, default=2, help="pydeps --max-module-depth (default: 2)")
    p.add_argument("--rankdir", choices=["TB", "BT", "LR", "RL"], default="LR", help="Graph direction")
    p.add_argument("--exclude-private", action="store_true", help="Hide private segments (._*); auto-disabled when focusing a private subpackage")
    p.add_argument("--noise-level", type=int, default=0, help="Prune nodes with degree > N (0 disables)")
    p.add_argument("--include-missing", action="store_true", help="Include modules not found on sys.path (workaround for rare cycles)")
    p.add_argument("--no-cluster", action="store_true", help="Disable clustering of external deps")
    p.add_argument("--no-only", action="store_true", help="Do not pass --only (show everything under target dir)")
    p.add_argument("--extra-exclude", action="append", default=[], help="Additional -x glob patterns (repeatable)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pydeps(
        src_root=args.src.resolve(),
        package=args.package,
        out=args.out.resolve(),
        subpath=args.subpath,
        depth=args.depth,
        rankdir=args.rankdir,
        exclude_private=args.exclude_private,
        extra_excludes=args.extra_exclude,
        noise_level=args.noise_level,
        include_missing=args.include_missing,
        cluster=not args.no_cluster,
        use_only=not args.no_only,
    )
    print("Done.")
    print(f"SVG: {args.out.resolve()}")


if __name__ == "__main__":
    main()

"""Usage
Try these exact commands

Full package (safe defaults):

python pydeps_command.py \
  --src ./src --package otaf \
  --out build/deps/otaf_full.svg \
  --depth 2 --rankdir LR

Focused private subpackage (won’t be filtered out):

python pydeps_command.py \
  --src ./src --package otaf --subpath _assembly_modeling \
  --out build/deps/assembly.svg \
  --depth 2 --rankdir LR

If you still hit that KeyError: 'otaf.surrogate', rerun with:

... --include-missing

and, if needed, without --only:

... --no-only

If anything else looks off, paste the output and I’ll tighten the flags.
"""
