from scorers.base_classification_scorer_factory import BaseClassificationScorerFactory
from scorers.result_scorer_accuracy import ResultScorerAccuracy


class ResultScorerAccuracyFactory(BaseClassificationScorerFactory):
    """
    Factory for accuracy results_scorer
    """

    def get(self):
        return ResultScorerAccuracy()
