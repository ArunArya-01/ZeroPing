
import time
from pathlib import Path
import subprocess
import sys

PARQUET = Path("featured_dataset.parquet")
ANALYZE_SCRIPT = Path("experiments/01_data_exploration/analyze_featured_dataset.py")

def get_size(path):
    try:
        return path.stat().st_size
    except:
        return 0

print("Waiting for the full featured_dataset.parquet to be written by the builder...")

last_size = 0
stable_count = 0
CHECK_INTERVAL = 15  # seconds

while True:
    if not PARQUET.exists():
        print(f"  [{time.strftime('%H:%M:%S')}] Parquet not found yet. Sleeping {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)
        continue

    current_size = get_size(PARQUET)
    print(f"  [{time.strftime('%H:%M:%S')}] Parquet size: {current_size:,} bytes")

    if current_size > last_size:
        stable_count = 0
        last_size = current_size
    else:
        stable_count += 1

    # Consider it done if size hasn't changed for ~3 checks (45s)
    if stable_count >= 3 and current_size > 100_000:  # basic sanity
        print("\nBuild appears complete (parquet size stable). Running analysis...")
        break

    time.sleep(CHECK_INTERVAL)

# Run the analyzer
try:
    result = subprocess.run(
        [sys.executable, str(ANALYZE_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=".",
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
except Exception as e:
    print(f"Failed to run analyzer: {e}")

print("\nAnalysis finished. You can also view FEATURED_DATASET.md for a full written explanation.")
