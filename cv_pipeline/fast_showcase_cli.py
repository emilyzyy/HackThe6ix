#!/usr/bin/env python3
"""Prepare a bounded early-exit confirmation showcase from a captured session."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.fast_showcase import run_fast_showcase
from pipeline.showcase_review import write_showcase_workspace


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", type=int, default=4)
    parser.add_argument("--min-safe", type=int, default=3)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--identify-limit", type=int, default=12)
    parser.add_argument(
        "--workspace", type=Path,
        help="write a disposable compact-confirmation workspace here",
    )
    parser.add_argument("--fixed-inventory", type=Path)
    parser.add_argument("--inventory-csv", type=Path)
    parser.add_argument("--generator-url")
    parser.add_argument(
        "--weights", type=Path,
        default=Path(__file__).resolve().parent / "models/lego_seg.pt",
    )
    args = parser.parse_args(argv)
    manifest = args.session / "multiview" / "manifest.json"
    output = args.output or (
        args.session / "analysis-runs" / "fast-showcase-demo"
    )
    path = run_fast_showcase(
        manifest,
        output,
        target=args.target,
        min_safe=args.min_safe,
        max_views=args.max_views,
        identify_limit=args.identify_limit,
        weights=args.weights,
        inventory_csv=args.inventory_csv,
    )
    print(f"SHOWCASE_READY {path}", flush=True)
    if args.workspace is not None:
        workspace = write_showcase_workspace(
            path,
            args.session,
            args.workspace,
            args.fixed_inventory or args.inventory_csv,
            generator_url=args.generator_url,
            inventory_csv=args.inventory_csv,
        )
        print(f"WORKSPACE_READY {workspace}", flush=True)


if __name__ == "__main__":
    main()
