from src.entity.artifact_entity import ClassificationMetricsArtifact
from src.exception import Customexception
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import sys


def get_classification_score(y_true, y_pred) -> ClassificationMetricsArtifact:
    try:
        # handle binary and multiclass
        average_method = "binary" if len(set(y_true)) <= 2 else "macro"

        model_f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
        model_precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        model_recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        model_accuracy = accuracy_score(y_true, y_pred)

        return ClassificationMetricsArtifact(
            f1_score=model_f1,
            precision_score=model_precision,
            recall_score=model_recall,
            accuracy_score=model_accuracy
        )

    except Exception as e:
        raise Customexception(e, sys)