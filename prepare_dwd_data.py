#!/usr/bin/env python3
"""Prepare consolidated ICON-D2 histories before running the forecast pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from pipeline.operational.config import load_operational_config
from pipeline.operational.prepare_dwd import (
    migrate_existing_processed,
    prepare_target_dwd,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-date",
        type=date.fromisoformat,
        default=date.today() + timedelta(days=1),
        help="Delivery date to prepare (default: tomorrow).",
    )
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        help="Only consolidate existing daily processed outputs.",
    )
    parser.add_argument(
        "--force-migration",
        action="store_true",
        help="Rebuild histories from all existing daily processed folders.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the 06 UTC input even if it verifies successfully.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing input even when DOWNLOAD_DWD=false in .env.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep consumed raw data after successful history verification.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parent
    config = load_operational_config(root)
    if args.download:
        config = replace(config, download_dwd=True)
    if args.migrate_only:
        counts = migrate_existing_processed(config, force=args.force_migration)
        print(
            "DWD migration complete: "
            + ", ".join(f"C={cluster}: {count}" for cluster, count in counts.items())
        )
        return 0
    prepare_target_dwd(
        config,
        args.target_date,
        force_download=args.force_download,
        force_migration=args.force_migration,
        delete_raw=(
            config.delete_dwd_raw_after_preprocess and not args.keep_raw
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
