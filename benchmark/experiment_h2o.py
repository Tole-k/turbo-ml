import logging
import h2o
from h2o.automl import H2OAutoML

from utils_old import BaseExperiment, Task

# Its not really working. Errors because number of rows is too small
class H2OExperiment(BaseExperiment):
    def __init__(self):
        h2o.init()
        self.name = "H2O"

    def find_best_model(self, dataset_path, task, duration, train_ratio=0.8):
        dataset = h2o.import_file(dataset_path)
        train, test = dataset.split_frame([train_ratio])
        x = train.columns
        y = train.columns[-1]
        x.remove(y)
        if task is Task.BINARY:
            train[y] = train[y].asfactor()
            test[y] = test[y].asfactor()
        aml = H2OAutoML(max_runtime_secs_per_model=duration, sort_metric="accuracy")
        aml.train(x=x, y=y, training_frame=train)
        lb = aml.leaderboard
        for model in lb["model_id"]:
            if model := self.find_model_in_string(model):
                print(model)
                return model


if __name__ == "__main__":
    experiment = H2OExperiment()
    experiment.perform_experiment()