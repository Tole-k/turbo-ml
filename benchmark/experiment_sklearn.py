import pandas as pd
import autosklearn.classification
from autosklearn.metrics import accuracy
from utils import BaseExperiment, Task, ClassificationFamily

FAMILY_MAPPING = {
    "adaboost": ClassificationFamily.BOOSTING,
    "bernoulli_nb": ClassificationFamily.BAYESIAN_METHOD,
    "decision_tree": ClassificationFamily.DECISION_TREE,
    "extra_trees": ClassificationFamily.RANDOM_FOREST, # similar to sklearn random forest
    "gaussian_nb": ClassificationFamily.BAYESIAN_METHOD,
    "gradient_boosting": ClassificationFamily.BOOSTING,
    "k_nearest_neighbors": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "lda": ClassificationFamily.DISCRIMINANT_ANALYSIS,
    "liblinear_svc": ClassificationFamily.SVM,
    "libsvm_svc": ClassificationFamily.SVM,
    "mlp": ClassificationFamily.NEURAL_NETWORK,
    "multinomial_nb": ClassificationFamily.BAYESIAN_METHOD,
    "passive_aggressive": ClassificationFamily.GENERALIZED_LINEAR_MODEL, # its in linear models category of sklearn
    "qda": ClassificationFamily.DISCRIMINANT_ANALYSIS,
    "random_forest": ClassificationFamily.RANDOM_FOREST,
    "sgd": ClassificationFamily.SVM # not sure, but by default it uses SVM according to the documentation
}

class AutoSklearnExperiment(BaseExperiment):
    def rank_families(self, dataset: pd.DataFrame, _: str, task: Task, seed, duration: int):
        if task is not Task.BINARY and task is not Task.MULTICLASS:
            raise NotImplementedError("Non classification task is not implemented") 
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=duration, metric=accuracy, seed=seed)
        automl.fit(X, y)
        leaderboard = automl.leaderboard(ensemble_only=False)
        ranked_families = []
        for model in leaderboard["type"]:
            if family := FAMILY_MAPPING.get(model):
                if family not in ranked_families:
                    ranked_families.append(family)
        return ranked_families
    
if __name__ == "__main__":
    experiment = AutoSklearnExperiment()
    experiment.perform_experiments()