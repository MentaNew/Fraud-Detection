from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
CHECKPOINTS_DIR = ROOT / "checkpoints"
METADATA_PATH = CHECKPOINTS_DIR / "metadata.json"
