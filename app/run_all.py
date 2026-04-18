"""
Unified pipeline orchestrator.

Validates environment, runs all data collection and processing scripts,
trains both models, then launches the unified Streamlit dashboard.

Data-collection steps are skipped when their expected output already exists,
so the script doesn't re-hit the CFBD API on repeat runs. Pass --force (or
--refresh-data) to re-fetch everything.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependency: python-dotenv")
    print("Install it with: pip install python-dotenv")
    sys.exit(1)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Latest season we expect to have data for. Used for sentinel files in
# per-year output directories (rosters, merged_rosters).
sys.path.insert(0, str(APP_DIR))
import config
LATEST_SEASON = config.season_end

# Each entry: (script filename, [expected output paths]).
# A script is skipped when ALL of its expected outputs exist and are non-empty.
COLLECTION_STEPS = [
    ("get_recruit_data.py",       [DATA_DIR / "recruit_data.csv"]),
    ("get_roster_data.py",        [DATA_DIR / "rosters" / f"{LATEST_SEASON}_rosters.csv"]),
    ("get_game_results.py",       [DATA_DIR / "game_results.csv"]),
    ("get_advanced_metrics.py",   [DATA_DIR / "advanced_metrics.csv"]),
    ("get_coaches.py",            [DATA_DIR / "coaches.csv"]),
    ("get_transfer_portal.py",    [DATA_DIR / "transfer_portal.csv"]),
    ("get_position_recruiting.py", [DATA_DIR / "position_recruiting.csv"]),
    ("get_team_locations.py",     [DATA_DIR / "team_locations.csv"]),
]

PROCESSING_STEPS = [
    ("merge_roster_rankings.py",  [DATA_DIR / "merged_rosters" / f"{LATEST_SEASON}_rosters.csv"]),
    ("get_team_starters.py",      [DATA_DIR / "starters_by_season.csv"]),
    ("build_matchup_features.py", [DATA_DIR / "Xy_train.csv"]),
    ("clean_dashboard_data.py",   [DATA_DIR / "dashboard_df.csv", DATA_DIR / "recruit_df.csv", DATA_DIR / "teams_df.csv", DATA_DIR / "pre_post_nil_df.csv"]),
]

# Analysis steps are cheap and consume the data, so always run.
ANALYSIS_STEPS = [
    ("generate_power_rankings.py", []),
    ("build_win_model.py", []),
]


def outputs_present(paths):
    """True if all paths exist and are non-empty."""
    return bool(paths) and all(p.exists() and p.stat().st_size > 0 for p in paths)


def run_step(script_path):
    print(f"\n{'=' * 70}")
    print(f"Running: {script_path.name}")
    print(f"{'=' * 70}\n")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script_path.name} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def execute(steps, force, label):
    """Run each step unless its outputs already exist (and not --force)."""
    for name, outputs in steps:
        script_path = APP_DIR / name
        if not script_path.exists():
            print(f"ERROR: missing script {script_path}")
            sys.exit(1)

        if not force and outputs and outputs_present(outputs):
            existing = ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in outputs)
            print(f"SKIP [{label}] {name} -- outputs present ({existing})")
            continue

        run_step(script_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full data + analysis pipeline, then launch the dashboard."
    )
    parser.add_argument(
        "--force", "--refresh-data", dest="force", action="store_true",
        help="Re-run data collection + processing even if outputs already exist.",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip launching the Streamlit dashboard at the end.",
    )
    args = parser.parse_args()

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        sys.exit(1)

    load_dotenv(env_path)
    if not os.getenv("CFBD_API_KEY", "").strip():
        print("ERROR: CFBD_API_KEY is missing or empty in .env")
        sys.exit(1)

    print("CFBD_API_KEY check passed. Starting pipeline...")
    if args.force:
        print("--force: re-fetching all data even if cached outputs exist.\n")
    else:
        print("(pass --force to re-fetch all data)\n")

    execute(COLLECTION_STEPS, args.force, "collect")
    execute(PROCESSING_STEPS, args.force, "process")
    execute(ANALYSIS_STEPS, force=True, label="analyze")  # always re-run analysis

    print(f"\n{'=' * 70}")
    print("Pipeline complete." + ("" if args.no_dashboard else " Launching dashboard..."))
    print(f"{'=' * 70}\n")

    if args.no_dashboard:
        return

    dashboard = APP_DIR / "streamlit_app.py"
    os.execvp("streamlit", ["streamlit", "run", str(dashboard)])


if __name__ == "__main__":
    main()
