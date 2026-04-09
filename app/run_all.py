"""
Unified pipeline orchestrator.

Validates environment, runs all data collection and processing scripts,
trains both models, then launches the unified Streamlit dashboard.
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


def run_step(script_path):
    print(f"\n{'=' * 70}")
    print(f"Running: {script_path.name}")
    print(f"{'=' * 70}\n")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script_path.name} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        sys.exit(1)

    load_dotenv(env_path)
    if not os.getenv("CFBD_API_KEY", "").strip():
        print("ERROR: CFBD_API_KEY is missing or empty in .env")
        sys.exit(1)

    # Phase 1: Data collection (order doesn't matter)
    collection_scripts = [
        APP_DIR / "get_recruit_data.py",
        APP_DIR / "get_roster_data.py",
        APP_DIR / "get_game_results.py",
        APP_DIR / "get_advanced_metrics.py",
        APP_DIR / "get_coaches.py",
        APP_DIR / "get_transfer_portal.py",
        APP_DIR / "get_position_recruiting.py",
        APP_DIR / "get_team_locations.py",
    ]

    # Phase 2: Data processing (order matters)
    processing_scripts = [
        APP_DIR / "merge_roster_rankings.py",
        APP_DIR / "get_team_starters.py",
        APP_DIR / "build_matchup_features.py",
    ]

    # Phase 3: Analysis
    analysis_scripts = [
        APP_DIR / "generate_power_rankings.py",
        APP_DIR / "build_win_model.py",
    ]

    all_scripts = collection_scripts + processing_scripts + analysis_scripts

    missing = [str(p) for p in all_scripts if not p.exists()]
    if missing:
        print("ERROR: Missing scripts:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("CFBD_API_KEY check passed. Starting pipeline...\n")

    for script in all_scripts:
        run_step(script)

    print(f"\n{'=' * 70}")
    print("Pipeline complete. Launching dashboard...")
    print(f"{'=' * 70}\n")

    dashboard = APP_DIR / "streamlit_app.py"
    os.execvp("streamlit", ["streamlit", "run", str(dashboard)])


if __name__ == "__main__":
    main()
