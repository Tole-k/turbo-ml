import sys
import logging
from typing import List
import pandas as pd
from evalml.automl import AutoMLSearch

from utils import BaseExperiment, Task, ClassificationFamily

FAMILIES_MAPPING = {
    "Baseline": ClassificationFamily.OTHER_METHOD,
    "CatBoost Classifier": ClassificationFamily.BOOSTING,
    "Decision Tree Classifier": ClassificationFamily.DECISION_TREE,
    "Elastic Net Classifier": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "XGBoost Classifier": ClassificationFamily.BOOSTING,
    "Extra Trees Classifier": ClassificationFamily.RANDOM_FOREST,
    "KNN Classifier": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "LightGBM Classifier": ClassificationFamily.BOOSTING,
    "Logistic Regression Classifier": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "Random Forest Classifier": ClassificationFamily.RANDOM_FOREST,
    "SVM Classifier": ClassificationFamily.SVM,
}


class EvalMlExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        self.task_mapping = {
            Task.MULTICLASS: "multiclass", Task.BINARY: "binary"}

    def find_family_in_string(self, string: str) -> ClassificationFamily:
        for model, family in FAMILIES_MAPPING.items():
            if model in string:
                return family
        logging.warning(f"Model not found in string: {string}")
        return None

    def rank_families(self, dataset: pd.DataFrame, _, task: Task, seed, duration: int) -> List[ClassificationFamily]:
        if task is not Task.BINARY and task is not Task.MULTICLASS:
            raise NotImplementedError(
                "Non classification task is not implemented")
        task = self.task_mapping[task]
        last_col = dataset.columns[-1]
        y = dataset[last_col]
        X = dataset.drop(last_col, axis=1)
        # cant choose accuracy as eval metric
        automl = AutoMLSearch(
            X_train=X, y_train=y, problem_type=task, max_time=duration, random_seed=seed)
        try:
            automl.search()
            families = []
            for pipeline in automl.rankings['pipeline_name']:
                print(pipeline)
                if family := self.find_family_in_string(pipeline):
                    families.append(family)
        except:
            families = ['', '', '', '', '']
        return families


if __name__ == "__main__":
    experiment = EvalMlExperiment()
    experiment.perform_experiments([int(sys.argv[1])])
