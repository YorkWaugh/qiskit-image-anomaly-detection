import numpy as np


class ClassicalPostprocessor:
    """Handles classical post-processing of anomaly scores.
    Currently, this involves applying a threshold to determine anomalies.
    """

    def __init__(self, k_std_dev=2.5, fallback_threshold=0.1):
        self.k_std_dev = k_std_dev
        self.fallback_threshold = fallback_threshold
        self.learned_threshold = None
        self.mean_normal_score_ = None
        self.std_normal_score_ = None
        print(
            f"ClassicalPostprocessor initialized with k_std_dev={k_std_dev}, fallback_threshold={fallback_threshold}"
        )

    def fit(self, normal_scores):
        if normal_scores is None or len(normal_scores) == 0:
            print(
                "Warning: No normal scores provided to ClassicalPostprocessor.fit(). Will use fallback threshold."
            )
            self.learned_threshold = None  # Explicitly mark as not learned
            self.mean_normal_score_ = None
            self.std_normal_score_ = None
            return

        normal_scores_np = np.array(normal_scores)
        if not np.all(np.isfinite(normal_scores_np)):
            print(
                "Warning: Non-finite values (NaN or Inf) found in normal_scores. Will use fallback threshold."
            )
            self.learned_threshold = None
            self.mean_normal_score_ = None
            self.std_normal_score_ = None
            return

        mean_score = np.mean(normal_scores_np)
        std_score = np.std(normal_scores_np)

        self.mean_normal_score_ = mean_score
        self.std_normal_score_ = std_score

        if std_score < 1e-6:  # Effectively zero or very small std dev
            print(
                f"Warning: Standard deviation of normal scores is very small ({std_score:.4f})."
            )
            # Threshold slightly above the mean if all normal scores are almost identical.
            self.learned_threshold = mean_score + 1e-5
            print(
                f"ClassicalPostprocessor: Learned threshold set to mean + epsilon = {self.learned_threshold:.4f}"
            )
        else:
            self.learned_threshold = mean_score + self.k_std_dev * std_score
            print(
                f"ClassicalPostprocessor: Learned threshold = {self.learned_threshold:.4f} (mean={mean_score:.4f}, std={std_score:.4f}, k={self.k_std_dev})"
            )

    def process_scores(self, anomaly_scores):
        active_threshold = self.fallback_threshold
        threshold_source = "fallback"

        if self.learned_threshold is not None:
            active_threshold = self.learned_threshold
            threshold_source = "learned"
        else:
            print(
                f"ClassicalPostprocessor: learned_threshold is None. Using fallback_threshold: {self.fallback_threshold}"
            )

        print(
            f"ClassicalPostprocessor: Using {threshold_source} threshold: {active_threshold:.4f}"
        )

        if not isinstance(anomaly_scores, np.ndarray):
            anomaly_scores = np.array(anomaly_scores)

        return anomaly_scores > active_threshold

    def get_threshold_info(self):
        if self.learned_threshold is not None:
            return {
                "type": "learned",
                "value": self.learned_threshold,
                "mean_normal_score": self.mean_normal_score_,
                "std_normal_score": self.std_normal_score_,
                "k_std_dev": self.k_std_dev,
            }
        return {"type": "fallback", "value": self.fallback_threshold}
