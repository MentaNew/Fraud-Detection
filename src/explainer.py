import logging

import numpy as np
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class SHAPExplainer:
    """SHAP TreeExplainer wrapper for the sklearn Pipeline (scaler → LightGBM).

    Extracts the scaler and the tree model separately so that TreeExplainer
    receives already-scaled data, which is required for exact Shapley values.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._scaler = pipeline.named_steps["scaler"]
        self._tree_explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        logger.info("SHAPExplainer initialised")

    def explain(self, X: np.ndarray, top_k: int = 10) -> list[list[dict]]:
        """Return top-k SHAP feature contributions for every row in X.

        Returns a list (one entry per sample) of sorted contribution dicts.
        """
        X_scaled = self._scaler.transform(X)
        raw = self._tree_explainer.shap_values(X_scaled)

        # Older SHAP (<0.42) returns [neg_class_vals, pos_class_vals];
        # newer SHAP returns a single array for the positive class directly.
        vals: np.ndarray = raw[1] if isinstance(raw, list) else raw

        results: list[list[dict]] = []
        for row in vals:
            top_idx = np.argsort(np.abs(row))[-top_k:][::-1]
            results.append(
                [
                    {
                        "feature": FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}",
                        "shap_value": float(row[i]),
                    }
                    for i in top_idx
                ]
            )
        return results
