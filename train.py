from src.data_pipeline import load_data, get_splits, preprocess_data
from src.model_training import train_baseline

def main():
    # 1. Load data
    path = "/Users/elmehdielkasmi/Desktop/ML_DL_projects/fraud_detection/creditcard.csv"
    X, y = load_data(path)

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = get_splits(X, y)

    # 3. Scale features
    #X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # 4. Train baseline
    model, auc = train_baseline(X_train, y_train, X_test, y_test)
    print(f"\n✅ Baseline training complete. AUC = {auc:.4f}")

if __name__ == "__main__":
    main()