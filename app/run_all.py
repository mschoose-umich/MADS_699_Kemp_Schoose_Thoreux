"""
Unified pipeline orchestrator.

Validates environment, runs all data collection and processing scripts,
trains both models, then launches the unified Streamlit dashboard.

Skips data collection scripts if their output files already exist.
Use --force to re-fetch all data.
"""

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


def run_step(script_path):
    print(f"\n{'=' * 70}")
    print(f"Running: {script_path.name}")
    print(f"{'=' * 70}\n")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script_path.name} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def outputs_exist(file_list):
    """Check if all expected output files exist and are non-empty."""
    for f in file_list:
        p = DATA_DIR / f
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def main():
    force = "--force" in sys.argv

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        sys.exit(1)

    load_dotenv(env_path)
    if not os.getenv("CFBD_API_KEY", "").strip():
        print("ERROR: CFBD_API_KEY is missing or empty in .env")
        sys.exit(1)

    # Phase 1: Data collection — each script mapped to its output file(s)
    collection_steps = [
        (APP_DIR / "get_recruit_data.py",       ["recruit_data.csv"]),
        (APP_DIR / "get_roster_data.py",        [f"rosters/{y}_rosters.csv" for y in range(2015, 2026)]),
        (APP_DIR / "get_game_results.py",       ["game_results.csv"]),
        (APP_DIR / "get_advanced_metrics.py",   ["advanced_metrics.csv"]),
        (APP_DIR / "get_coaches.py",            ["coaches.csv"]),
        (APP_DIR / "get_transfer_portal.py",    ["transfer_portal.csv"]),
        (APP_DIR / "get_position_recruiting.py", ["position_recruiting.csv"]),
        (APP_DIR / "get_team_locations.py",     ["team_locations.csv"]),
    ]

    # Phase 2: Data processing (order matters)
    processing_steps = [
        (APP_DIR / "merge_roster_rankings.py",  [f"merged_rosters/{y}_rosters.csv" for y in range(2015, 2026)]),
        (APP_DIR / "get_team_starters.py",      ["starters_by_season.csv"]),
        (APP_DIR / "build_matchup_features.py", ["Xy_train.csv", "2025_games_clean.csv", "2025_rosters.csv"]),
        (APP_DIR / "clean_dashboard_data.py",   ["dashboard_df.csv", "recruit_df.csv", "teams_df.csv", "pre_post_nil_df.csv"]),
    ]

    # Phase 3: Analysis
    analysis_steps = [
        (APP_DIR / "generate_power_rankings.py", ["power_rankings_national.csv", "power_rankings_predictions.csv", "model_metrics.json"]),
        (APP_DIR / "build_win_model.py",         ["production_model.pkl"]),
    ]

    all_steps = collection_steps + processing_steps + analysis_steps

    missing_scripts = [str(s) for s, _ in all_steps if not s.exists()]
    if missing_scripts:
        print("ERROR: Missing scripts:")
        for m in missing_scripts:
            print(f"  - {m}")
        sys.exit(1)

    print("CFBD_API_KEY check passed. Starting pipeline...\n")

    for script, expected_outputs in all_steps:
        if not force and outputs_exist(expected_outputs):
            print(f"Skipping {script.name} — output files already exist")
            continue
        run_step(script)

    print(f"\n{'=' * 70}")
    print("Pipeline complete. Launching dashboard...")
    print(f"{'=' * 70}\n")

    dashboard = APP_DIR / "streamlit_app.py"
    os.execvp("streamlit", ["streamlit", "run", str(dashboard)])


if __name__ == "__main__":
    main()
