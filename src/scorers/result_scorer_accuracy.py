from scorers.base_classification_scorer import BaseClassificationScorer
from sklearn.metrics import accuracy_score

import numpy as np


class ResultScorerAccuracy(BaseClassificationScorer):
    """
    Calculate Accuracy
    """

    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        y_pred = np.array(y_pred)
        y_actual = np.array(y_actual)
        print(y_pred)
        print(y_actual)

        # if 2 D array, get max label index
        if len(y_actual.shape) == 2:
            y_actual = y_actual.reshape(-1)
            y_pred = y_pred.reshape(-1)
        accu = accuracy_score(y_actual, y_pred)

        return accu
