import pandas as pd
import autosklearn.classification
from autosklearn.metrics import accuracy
# from sklearn.model_selection import train_test_split
from utils import BaseExperiment, Task
ALGORITHM_FAMILY_MAPPING = {
    "adaboost":
    "bernoulli_nb":
    "decision_tree":
    "extra_trees":
    "gaussian_nb":
    "gradient_boosting":
    "k_nearest_neighbors":
    "lda":
    "liblinear_svc":
    "libsvm_svc":
    "mlp":
    "multinomial_nb":
    "passive_aggressive":
    "qda":
    "random_forest":
    "sgd":
}

class AutoSklearnExperiment(BaseExperiment):
    def __init__(self):
        self.name = "AutoSklearn"

    def find_best_model(self, dataset_path, task, duration, train_ratio=0.8):
        dataset = pd.read_csv(dataset_path)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        # It doesnt take test data into account
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio)
        if task is not Task.BINARY and task is not Task.MULTICLASS:
            raise NotImplementedError("Non classification task is not implemented")
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=duration, metric=accuracy)
        automl.fit(X, y)
        leaderboard = automl.leaderboard(ensemble_only=False)
        print(leaderboard)
        for model in leaderboard["type"]:
            # it uses snake case
            model = model.replace("_", "")
            if model := self.find_model_in_string(model):
                print(model)
                return model
        return automl
    
if __name__ == "__main__":
    experiment = AutoSklearnExperiment()
    experiment.perform_experiment()