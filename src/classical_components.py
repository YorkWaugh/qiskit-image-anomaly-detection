import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class ClassicalPostprocessor:
    """Handles classical post-processing of anomaly scores using a classifier."""

    def __init__(self, random_state=42):
        self.classifier = SVC(
            random_state=random_state, class_weight="balanced", probability=True
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        print(f"ClassicalPostprocessor initialized with SVC.")

    def fit(self, scores, labels):
        """
        Fits the classifier using the provided scores and labels.
        Scores: array-like, shape (n_samples,)
        Labels: array-like, shape (n_samples,), 0 for normal, 1 for anomaly
        """
        if scores is None or len(scores) == 0:
            print(
                "Warning: No scores provided to ClassicalPostprocessor.fit(). Classifier not fitted."
            )
            self.is_fitted = False
            return

        if labels is None or len(labels) == 0:
            print(
                "Warning: No labels provided to ClassicalPostprocessor.fit(). Classifier not fitted."
            )
            self.is_fitted = False
            return

        if len(scores) != len(labels):
            print(
                "Warning: Mismatch between number of scores and labels. Classifier not fitted."
            )
            self.is_fitted = False
            return

        scores_np = np.array(scores).reshape(-1, 1)  # Reshape for scaler and classifier
        labels_np = np.array(labels)

        if not np.all(np.isfinite(scores_np)):
            print(
                "Warning: Non-finite values (NaN or Inf) found in scores. Classifier not fitted."
            )
            self.is_fitted = False
            return

        if len(np.unique(labels_np)) < 2:
            print(
                f"Warning: Only one class present in labels: {np.unique(labels_np)}. Classifier requires at least two classes. Model not fitted."
            )
            self.is_fitted = False
            return

        try:
            self.scaler.fit(scores_np)
            scaled_scores = self.scaler.transform(scores_np)
            self.classifier.fit(scaled_scores, labels_np)
            self.is_fitted = True
            print(f"ClassicalPostprocessor: SVC classifier fitted successfully.")
        except Exception as e:
            print(f"Error during classifier fitting: {e}")
            self.is_fitted = False

    def process_scores(self, anomaly_scores):
        """
        Predicts labels for the given anomaly scores using the fitted classifier.
        Returns an array of predicted labels (0 or 1).
        """
        if not self.is_fitted:
            print(
                "Error: Classifier not fitted. Call fit() first. Returning empty predictions."
            )
            print(
                "Warning: Classifier not fitted. Predicting all as normal (0) as a fallback."
            )
            return np.zeros(len(anomaly_scores), dtype=int)

        if not isinstance(anomaly_scores, np.ndarray):
            anomaly_scores = np.array(anomaly_scores)

        if anomaly_scores.ndim == 1:
            anomaly_scores_reshaped = anomaly_scores.reshape(-1, 1)
        else:
            anomaly_scores_reshaped = anomaly_scores

        if not np.all(np.isfinite(anomaly_scores_reshaped)):
            print(
                "Warning: Non-finite values (NaN or Inf) found in anomaly_scores for prediction. Returning empty predictions for these."
            )
            predictions = np.zeros(len(anomaly_scores_reshaped), dtype=int)
            finite_mask = np.all(np.isfinite(anomaly_scores_reshaped), axis=1)
            if np.any(finite_mask):
                scaled_scores = self.scaler.transform(
                    anomaly_scores_reshaped[finite_mask]
                )
                predictions[finite_mask] = self.classifier.predict(scaled_scores)
            return predictions

        try:
            scaled_scores = self.scaler.transform(anomaly_scores_reshaped)
            predictions = self.classifier.predict(scaled_scores)
            print(f"ClassicalPostprocessor: Predictions made using SVC.")
            return predictions
        except NotFittedError:
            print(
                "Error: Scaler or Classifier not fitted. Call fit() first. Returning empty predictions."
            )
            return np.zeros(len(anomaly_scores_reshaped), dtype=int)
        except Exception as e:
            print(f"Error during score processing: {e}")
            return np.zeros(len(anomaly_scores_reshaped), dtype=int)

    def get_classifier_info(self):  # Renamed from get_threshold_info
        if self.is_fitted:
            return {
                "type": "classifier",
                "model": self.classifier.__class__.__name__,
                "params": self.classifier.get_params(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
            }
        return {"type": "unfitted", "model": self.classifier.__class__.__name__}
