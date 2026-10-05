"""Common paths for the local UnmixingMaldi code tree."""

from pathlib import Path

from helpers.local_env import load_local_env


load_local_env()


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed_h5"
CONFIG_DIR = CODE_DIR / "configs"
MIXTURES_DIR = DATA_DIR / "mixtures"
RESULTS_DIR = REPO_ROOT / "evaluation_results"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
