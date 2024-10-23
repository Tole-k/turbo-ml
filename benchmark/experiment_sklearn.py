import pandas as pd
import autosklearn.classification
from autosklearn.metrics import accuracy
# from sklearn.model_selection import train_test_split
from utils import BaseExperiment

class AutoSklearnExperiment(BaseExperiment):
    def __init__(self):
        self.name = "AutoSklearn"
        self.task_mapping = {"multiclass_classification": "classification", "binary_classification": "classification"}

    def find_best_model(self, dataset_path, task, duration, train_ratio=0.8):
        dataset = pd.read_csv(dataset_path)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        # It doesnt take test data into account
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio)
        if task != "classification":
            return None
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