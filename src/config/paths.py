from pathlib import Path

# Resolve from this file, independently of the script or notebook working directory.
ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

CSV_DIR = RESULTS_DIR / "csvs"
IMAGE_DIR = RESULTS_DIR / "images" 
PLOT_DIR = RESULTS_DIR / "plots" 
MODEL_DIR = RESULTS_DIR / "models"
WEIGHTS_DIR = MODEL_DIR / "weights"
