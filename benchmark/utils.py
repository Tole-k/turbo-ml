import os
import abc
import csv
import json
from collections import defaultdict
from datetime import datetime

TRAIN_RATIO = 0.8
TOP_N = [1, 3, 5, 10]
TEST_DURATIONS = [60, 15 * 60, 60 * 60, 2 * 60 * 60]
MODEL_NAMES = [
    "NeuralNetworkModel",
    "XGBoostClassifier",
    "AdaBoostClassifier",
    "BaggingClassifier",
    "BernoulliNB",
    "CalibratedClassifierCV",
    "CategoricalNB",
    "ComplementNB",
    "DecisionTreeClassifier",
    "DummyClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "GaussianNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "KNeighborsClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "LinearDiscriminantAnalysis",
    "LinearSVC",
    "LogisticRegression",
    "LogisticRegressionCV",
    "MLPClassifier",
    "MultinomialNB",
    "NearestCentroid",
    "NuSVC",
    "PassiveAggressiveClassifier",
    "Perceptron",
    "QuadraticDiscriminantAnalysis",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SVC",
]

AVAILABLE_TASKS = ["multiclass_classification", "binary_classification"]

class BaseExperiment(abc.ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.task_mapping = {"multiclass_classification": "multiclass_classification", "binary_classification": "binary_classification"}
    
    @abc.abstractmethod
    def find_best_model(self, dataset_path, task, duration, train_ration=0.8):
        pass
        
        
    def perform_experiment(self):
        datasets_info = self.__load_datasets_info()
        datasets = self.__get_datasets()
        results = defaultdict(lambda: defaultdict(float))
        for dataset_path in datasets:
            dataset_name = dataset_path.replace("benchmark/datasets/", "").replace(".csv", "")
            task = datasets_info[dataset_name]["task_detailed"] 
            best_models_sorted = sorted(MODEL_NAMES, key=lambda x: datasets_info[dataset_name][x], reverse=True)
            for duration in TEST_DURATIONS:
                best_model = self.find_best_model(dataset_path, self.task_mapping[task], duration, TRAIN_RATIO)
                for top_n in TOP_N:
                    results[duration][top_n] += 1/len(datasets) if best_model in best_models_sorted[:top_n] else 0
        self.__save_to_json(self.name, results)

    def find_model_in_string(self, string):
        for model in MODEL_NAMES:
            if model.lower() in string.lower().replace(" ", ""):
                return model
        return None

    def __get_datasets(self):
        return ["benchmark/datasets/iris.csv"]


    def __save_to_json(self, library_name: str, data: dict):
        output_path = f"./benchmark/outputs/{library_name}-{datetime.now()}.json"
        if not os.path.exists("benchmark/outputs"):
            os.makedirs("benchmark/outputs")
        with open(output_path, "w") as f:
            json.dump(data, f)


    def __load_datasets_info(self):
        data = {}
        path = "benchmark/datasets/test.csv"
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                del row["name"]
                data[name] = row
        return data
