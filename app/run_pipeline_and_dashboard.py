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


def run_step(script_path: Path) -> None:
    print(f"\n{'=' * 80}")
    print(f"Running: {script_path.name}")
    print(f"{'=' * 80}\n")

    result = subprocess.run([sys.executable, str(script_path)], check=False)

    if result.returncode != 0:
        print(f"\nERROR: {script_path.name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parent
    env_path = project_root / ".env"

    if not env_path.exists():
        print("ERROR: .env file not found in project root.")
        print(f"Expected location: {env_path}")
        sys.exit(1)

    load_dotenv(env_path)

    cfbd_api_key = os.getenv("CFBD_API_KEY")
    if not cfbd_api_key or not cfbd_api_key.strip():
        print("ERROR: CFBD_API_KEY is missing or empty in .env")
        print("Add a line like this to your .env file:")
        print("CFBD_API_KEY=your_api_key_here")
        sys.exit(1)

    scripts_in_order = [
        app_dir / "get_game_data.py",
        app_dir / "get_recruit_data.py",
        app_dir / "get_roster_data.py",
        app_dir / "merge_roster_rankings.py",
        app_dir / "team_starters_pipe.py",
        app_dir / "data_clean_and_split.py",
        app_dir / "build_model.py",
    ]

    missing_scripts = [str(p) for p in scripts_in_order if not p.exists()]
    if missing_scripts:
        print("ERROR: The following required scripts were not found:")
        for path in missing_scripts:
            print(f"  - {path}")
        sys.exit(1)

    print("CFBD_API_KEY check passed.")
    print("Starting pipeline...")

    for script in scripts_in_order:
        run_step(script)

    print(f"\n{'=' * 80}")
    print("Pipeline completed successfully.")
    print("Launching Streamlit app...")
    print(f"{'=' * 80}\n")

    streamlit_app = app_dir / "streamlit_app.py"
    if not streamlit_app.exists():
        print(f"ERROR: Streamlit app not found: {streamlit_app}")
        sys.exit(1)

    os.execvp(
        "streamlit",
        ["streamlit", "run", str(streamlit_app)]
    )


if __name__ == "__main__":
    main()