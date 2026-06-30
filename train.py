import argparse

from src.config import DATA_PATH
from src.data_pipeline import get_splits, load_data
from src.model_training import train_baseline
from src.utils import setup_logging


def main(data_path=DATA_PATH):
    setup_logging()
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = get_splits(X, y)
    _, auc = train_baseline(X_train, y_train, X_test, y_test)
    print(f"\nBaseline training complete. AUC = {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline fraud detection model")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to creditcard.csv")
    args = parser.parse_args()
    main(args.data)
