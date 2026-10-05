#!/usr/bin/env python3
"""Run every variant declared in a Transformer ablation YAML."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR_LOCAL = Path(__file__).resolve().parents[1]
if str(CODE_DIR_LOCAL) not in sys.path:
    sys.path.insert(0, str(CODE_DIR_LOCAL))

import yaml

from helpers.local_env import expand_config_values, load_local_env
from helpers.paths import CODE_DIR, CONFIG_DIR, PROJECT_ROOT
from train_transformer import resolve_config_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Ablation YAML with ablation.variants.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve each variant without training.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    load_local_env()
    with config_path.open() as handle:
        config = expand_config_values(yaml.safe_load(handle) or {})
    variants = (config.get("ablation") or {}).get("variants") or {}
    if not variants:
        raise ValueError(f"No ablation.variants found in {config_path}")

    train_script = Path(__file__).resolve().with_name("train_transformer.py")
    for variant in variants:
        command = [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
            "--variant",
            str(variant),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(f"\n=== Transformer ablation: {config['ablation'].get('name')} / {variant} ===", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
