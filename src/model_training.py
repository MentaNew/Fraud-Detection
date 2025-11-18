import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_baseline(X_train, y_train, X_test, y_test, save_path="checkpoints/baseline.pkl"):
    """
    Train a simple LightGBM baseline model and evaluate it.
    Saves the model to `save_path` and returns the model and AUC score.
    """
    scaler = StandardScaler()
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42
    )

    pipe = Pipeline([
        ("scaler", scaler), 
        ("model", model)

    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    print(f"\n✅ ROC-AUC: {auc:.4f}\n")
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(pipe, save_path)
    print(f"💾 Model saved to {save_path}")
    return pipe, auc