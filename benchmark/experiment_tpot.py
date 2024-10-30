import os
from datetime import datetime
import logging
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from utils import BaseExperiment, Task

# It may have problem with ensemble models
class TPotExperiment(BaseExperiment):
    def __init__(self):
        self.name = "TPot"

    def __find_best_model_classification(self, dataset_path, duration, train_ratio=0.8):
        dataset = pd.read_csv(dataset_path)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio)
        pipeline_optimizer = TPOTClassifier(max_time_mins=duration/60, scoring="accuracy")
        pipeline_optimizer.fit(X_train, y_train)
        logging.info(pipeline_optimizer.score(X_test, y_test))
        output_folder = f'benchmark/TPot-outputs/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = f'{output_folder}tpot_exported_pipeline-{datetime.now()}.py'
        pipeline_optimizer.export(output_path)
        with open(output_path, "r") as f:
            content = f.read()
            model_name = self.find_model_in_string(content)
            if model_name is None:
                logging.warning(f"Model not found in {output_path}")
            return model_name


    def find_best_model(self, dataset_path, task, duration, train_ratio=0.8):
        if task is Task.BINARY or task is Task.MULTICLASS:
            return self.__find_best_model_classification(dataset_path, duration, train_ratio)
        else:
            raise NotImplementedError("Non classification task is not implemented")

if __name__ == "__main__":
    experiment = TPotExperiment()
    experiment.perform_experiment()