import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class ClassicalPostprocessor:
    """RandomForest-based postprocessor."""

    def __init__(self, random_state=42):
        # init classifier
        self.classifier = RandomForestClassifier(
            random_state=random_state, class_weight="balanced"
        )
        self.is_fitted = False

    def fit(self, scores, labels):
        """Train classifier on score-label pairs."""
        if scores is None or len(scores) == 0:
            self.is_fitted = False
            return

        if labels is None or len(labels) == 0:
            self.is_fitted = False
            return

        if len(scores) != len(labels):
            self.is_fitted = False
            return

        scores_np = np.array(scores).reshape(-1, 1)
        labels_np = np.array(labels)

        if not np.all(np.isfinite(scores_np)):
            self.is_fitted = False
            return

        if len(np.unique(labels_np)) < 2:
            self.is_fitted = False
            return

        try:
            # grid search for best params
            param_grid = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
            grid = GridSearchCV(
                estimator=self.classifier,
                param_grid=param_grid,
                cv=3,
                scoring="accuracy",
            )
            grid.fit(scores_np, labels_np)
            self.classifier = grid.best_estimator_
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def process_scores(self, anomaly_scores):
        """Predict labels from scores."""
        if not self.is_fitted:
            return np.zeros(len(anomaly_scores), dtype=int)

        if not isinstance(anomaly_scores, np.ndarray):
            anomaly_scores = np.array(anomaly_scores)

        if anomaly_scores.ndim == 1:
            anomaly_scores_reshaped = anomaly_scores.reshape(-1, 1)
        else:
            anomaly_scores_reshaped = anomaly_scores

        if not np.all(np.isfinite(anomaly_scores_reshaped)):
            predictions = np.zeros(len(anomaly_scores_reshaped), dtype=int)
            finite_mask = np.all(np.isfinite(anomaly_scores_reshaped), axis=1)
            if np.any(finite_mask):
                reshaped = anomaly_scores_reshaped[finite_mask]
                predictions[finite_mask] = self.classifier.predict(reshaped)
            return predictions

        try:
            # predict
            predictions = self.classifier.predict(anomaly_scores_reshaped)
            return predictions
        except Exception:
            return np.zeros(len(anomaly_scores_reshaped), dtype=int)

    def get_classifier_info(self):
        """Return model info or unfitted status."""
        if self.is_fitted:
            return {
                "type": "classifier",
                "model": self.classifier.__class__.__name__,
                "params": self.classifier.get_params(),
            }
        return {"type": "unfitted", "model": self.classifier.__class__.__name__}
