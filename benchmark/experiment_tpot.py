import os
from datetime import datetime
import logging
from typing import List
import pandas as pd
from tpot import TPOTClassifier
from utils import BaseExperiment, Task, ClassificationFamily


FAMILIES_MAPPING = {
    'GaussianNB': ClassificationFamily.BAYESIAN_METHOD,
    'BernoulliNB': ClassificationFamily.BAYESIAN_METHOD,
    'MultinomialNB': ClassificationFamily.BAYESIAN_METHOD, 
    'DecisionTreeClassifier': ClassificationFamily.DECISION_TREE,
    'ExtraTreesClassifier':  ClassificationFamily.RANDOM_FOREST, # same as in sklearn experiment
    'RandomForestClassifier': ClassificationFamily.RANDOM_FOREST, 
    'GradientBoostingClassifier': ClassificationFamily.BOOSTING,
    'KNeighborsClassifier': ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    'LinearSVC': ClassificationFamily.SVM,
    'LogisticRegression': ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    'XGBClassifier': ClassificationFamily.BOOSTING,
    'SGDClassifier': ClassificationFamily.SVM, # not sure, but by default it uses SVM according to the documentation
    'MLPClassifier': ClassificationFamily.NEURAL_NETWORK
}

# It may have problem with ensemble models
class TPotExperiment(BaseExperiment):
    def __init__(self):
        self.name = "TPot"

    def find_model_in_string(self, content: str) -> ClassificationFamily:
        for model_name, family in FAMILIES_MAPPING.items():
            if model_name in content:
                return family
        logging.warning(f"Model not found in {content}")
        return None

    def rank_families(self, dataset: pd.DataFrame, _, task: Task, seed, duration: int) -> List[ClassificationFamily]:
        if task is not Task.BINARY and task is not Task.MULTICLASS:
            raise NotImplementedError("Non classification task is not implemented") 
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        pipeline_optimizer = TPOTClassifier(max_time_mins=duration/60, scoring="accuracy", random_state=seed)
        pipeline_optimizer.fit(X, y)
        output_folder = f'benchmark/TPot-outputs/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = f'{output_folder}tpot_exported_pipeline-{datetime.now()}.py'
        pipeline_optimizer.export(output_path)
        with open(output_path, "r") as f:
            content = f.read()
            family = self.find_model_in_string(content)
            return [family]


if __name__ == "__main__":
    experiment = TPotExperiment()
    experiment.perform_experiments([0], [60])