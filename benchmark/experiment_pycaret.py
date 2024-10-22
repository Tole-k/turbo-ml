import logging
import pandas as pd
from pycaret.classification import ClassificationExperiment
from utils import BaseExperiment

class PycaretExperiment(BaseExperiment):
    def __init__(self):
        self.name = "Pycaret"
        self.task_mapping = {"multiclass_classification": "classification", "binary_classification": "classification"}
        
    def __find_best_model_classification(self, dataset_path, duration, train_ratio=0.8):
        data = pd.read_csv(dataset_path)
        s = ClassificationExperiment()
        s.setup(data, target=data.columns[-1], train_size=train_ratio)
        best_models = s.compare_models(n_select=10, budget_time=duration/60)
        best_accuracy_model = s.automl(optimize="Accuracy")
        if model_name := self.find_model_in_string(type(best_accuracy_model).__name__):
            return model_name
        logging.warning(f"Model {type(best_accuracy_model).__name__} not found in the list of models")
        for model in best_models:
            if model_name := self.find_model_in_string(type(model).__name__):
                return model_name

        
    def find_best_model(self, dataset_path, task, duration, train_ratio=0.8):
        if task == "classification":
            return self.__find_best_model_classification(dataset_path, duration, train_ratio)
        

if __name__ == "__main__":
    experiment = PycaretExperiment()
    experiment.perform_experiment()