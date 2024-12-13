import logging
from typing import List
import pandas as pd
from pycaret.classification import ClassificationExperiment
from utils import BaseExperiment, Task, ClassificationFamily


FAMILIES_MAPPING = {
    "LogisticRegression": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "KNeighborsClassifier": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "GaussianNB": ClassificationFamily.BAYESIAN_METHOD,
    "DecisionTreeClassifier": ClassificationFamily.DECISION_TREE,
    "SGDClassifier": ClassificationFamily.SVM,
    "SVC": ClassificationFamily.SVM,
    "GaussianProcessClassifier": ClassificationFamily.BAYESIAN_METHOD, # not sure
    "MLPClassifier": ClassificationFamily.NEURAL_NETWORK,
    "RidgeClassifier": ClassificationFamily.GENERALIZED_LINEAR_MODEL,
    "RandomForestClassifier": ClassificationFamily.RANDOM_FOREST,
    "QuadraticDiscriminantAnalysis": ClassificationFamily.DISCRIMINANT_ANALYSIS,
    "AdaBoostClassifier": ClassificationFamily.BOOSTING,
    "GradientBoostingClassifier": ClassificationFamily.BOOSTING,
    "LinearDiscriminantAnalysis": ClassificationFamily.DISCRIMINANT_ANALYSIS,
    "ExtraTreesClassifier": ClassificationFamily.RANDOM_FOREST,
    "XGBClassifier": ClassificationFamily.BOOSTING,
    "LGBMClassifier": ClassificationFamily.BOOSTING,
    "CatBoostClassifier": ClassificationFamily.BOOSTING,
    "DummyClassifier": ClassificationFamily.OTHER_METHOD,
}

# pycaret likes to use up more time than allocated
class PycaretExperiment(BaseExperiment):
    def rank_families(self, dataset: pd.DataFrame, _, task: Task, seed, duration: int) -> List[ClassificationFamily]:
        if task is not Task.BINARY and task is not Task.MULTICLASS:
            raise NotImplementedError("Non classification task is not implemented")
        s = ClassificationExperiment()
        s.setup(dataset, target=dataset.columns[-1], session_id=seed)
        best_models = s.compare_models(n_select=100, budget_time=duration/60)
        best_families = []
        for model in best_models:
            model_name = model.__class__.__name__
            if family := FAMILIES_MAPPING.get(model_name):
                best_families.append(family)
            else:
                logging.warning(f"Couldnt find mapping of {model_name} to a family")
        return best_families
        
import sys
if __name__ == "__main__":
    experiment = PycaretExperiment()
    experiment.perform_experiments([sys.argv[1]])