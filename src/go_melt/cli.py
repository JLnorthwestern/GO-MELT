#!/usr/bin/env python3
"""
Minimal CLI for GO-MELT.

Usage example:
  go-melt --gpu 0 --config examples/examples.json
"""

import os
import sys
import argparse
import logging

logger = logging.getLogger("go_melt.cli")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="go-melt",
        description="Run GO-MELT thermal solver",
        epilog="Usage example: go-melt --gpu 0 --config examples/examples.json",
    )

    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (e.g. 0). If omitted, uses default device.",
    )

    p.add_argument(
        "--config", type=str, required=True, help="Path to example/config JSON file"
    )

    p.add_argument(
        "--dry-run", action="store_true", help="Print resolved args and exit"
    )

    p.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v or -vv for more)",
    )

    return p.parse_args(argv)


def configure_logging(verbosity: int):
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def set_device_env(gpu_index: str | None):
    if gpu_index is None:
        return
    # common env var for CUDA visible devices; adjust if using different backend
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    logger.info("Set CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])


def run_package(config_path: str | None):
    """
    Call into the package entrypoint. Adjust imports if your run function has a
    different name. This code tries two common names to remain robust during refactor:
      - go_melt.core.go_melt(config_path)
      - go_melt.main(config_path)
    """
    try:
        from go_melt.core.go_melt import (
            go_melt as package_run,
        )  # former script -> function
    except Exception:
        try:
            from go_melt import main as package_run
        except Exception as exc:
            logger.exception("Could not import package run entrypoint: %s", exc)
            raise SystemExit(1)
    # call the package run function; prefer signature run(config_path)
    if config_path is None:
        logger.info("No config path provided; calling run() without args")
        return package_run()
    else:
        return package_run(config_path)


def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.verbose)
    set_device_env(args.gpu)

    if args.dry_run:
        print("dry-run:", {"gpu": args.gpu, "config": args.config})
        return 0

    if args.config is None:
        logger.warning(
            "No config file provided; continuing may require defaults inside the package"
        )

    try:
        run_package(args.config)
    except SystemExit as e:
        # let package exit codes propagate
        raise
    except Exception as e:
        logger.exception("Run failed: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
