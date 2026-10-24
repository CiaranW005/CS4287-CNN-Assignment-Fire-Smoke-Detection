from pathlib import Path

ROOT = Path.cwd().parents[1]

DATASET_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

CSV_DIR = RESULTS_DIR / "csvs"
IMAGE_DIR = RESULTS_DIR / "images" 
PLOT_DIR = RESULTS_DIR / "plots" 
MODEL_DIR = RESULTS_DIR / "models"