import numpy as np


class ObjectDetector:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        y_pred = np.empty(len(X), dtype=np.object)
        y_pred[:] = [[(1.0, 112.0, 112.0, 112.0)] for img in X]
        return y_pred
