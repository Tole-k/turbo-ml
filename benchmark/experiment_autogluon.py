from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn import model_selection
from utils_old import BaseExperiment, Task

class AutoGluonExperiment(BaseExperiment):
    def __init__(self):
        self.name = "Auto-gluon"
        self.task_mapping = {Task.MULTICLASS: "multiclass", Task.BINARY: "binary"}

    def find_best_model(self, dataset_path, task, duration, train_ratio=0.8):
        if task not in self.task_mapping:
            raise NotImplementedError("Non classification task is not implemented")
        task = self.task_mapping[task]
        dataset = TabularDataset(dataset_path)
        train_dataset, test_dataset = model_selection.train_test_split(dataset, train_size=train_ratio)
        predictor = TabularPredictor(
            label=dataset.columns[-1], path="AutoGluon-outputs", problem_type=task, eval_metric="accuracy"
        ).fit(train_dataset, time_limit=duration)
        for model in predictor.leaderboard(test_dataset)["model"]:
            if model := self.find_model_in_string(model):
                return model


if __name__ == "__main__":
    experiment = AutoGluonExperiment()
    experiment.perform_experiment()