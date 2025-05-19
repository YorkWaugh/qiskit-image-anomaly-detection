import numpy as np


class ClassicalPostprocessor:

    def __init__(self):
        pass

    def process_scores(self, anomaly_scores, threshold=0.5):
        if not isinstance(anomaly_scores, (list, np.ndarray)):
            raise TypeError("Input 'anomaly_scores' must be a list or numpy array.")

        scores_array = np.array(anomaly_scores)

        return scores_array > threshold
