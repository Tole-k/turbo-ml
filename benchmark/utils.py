import os
import abc
import json
import logging
from glob import glob
from enum import Enum, auto
from collections import defaultdict
from datetime import datetime

import pandas as pd

TEST_DURATIONS = [30, 60]


class ClassificationFamily(Enum):
    DISCRIMINANT_ANALYSIS = auto()
    BAYESIAN_METHOD = auto()
    NEURAL_NETWORK = auto()
    SVM = auto()
    DECISION_TREE = auto()
    RULE_BASED_METHOD = auto()
    BOOSTING = auto()
    BAGGING = auto()
    STACKING = auto()
    RANDOM_FOREST = auto()
    OTHER_ENSEMBLE = auto()
    GENERALIZED_LINEAR_MODEL = auto()
    NEAREST_NEIGHBOR_METHOD = auto()
    PARTIAL_LEAST_SQUARES_AND_PRINCIPAL_COMPONENT_REGRESSION = auto()
    LOGISTIC_AND_MULTINOMINAL_REGRESSION = auto()
    MUTIVARIATE_ADAPTIVE_REGRESSION_SPLINE = auto()
    OTHER_METHOD = auto()


_FAMILY_MAPPING = {
    "Bagging_(BAG)": ClassificationFamily.BAGGING,
    "Bayesian_Methods_(BY)": ClassificationFamily.BAYESIAN_METHOD,
    "Boosting_(BST)": ClassificationFamily.BOOSTING,
    "Decision_Trees_(DT)": ClassificationFamily.DECISION_TREE,
    "Discriminant_Analysis_(DA)": ClassificationFamily.DISCRIMINANT_ANALYSIS,
    "Generalized_Linear_Models_(GLM)": ClassificationFamily.GENERALIZED_LINEAR_MODEL,
    "Logistic_and_Multinomial_Regression_(LMR)": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "Multivariate_Adaptive_Regression_Splines_(MARS)": ClassificationFamily.MUTIVARIATE_ADAPTIVE_REGRESSION_SPLINE,
    "Nearest_Neighbor_Methods_(NN)": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "Neural_Networks_(NNET)": ClassificationFamily.NEURAL_NETWORK,
    "Other_Ensembles_(OEN)": ClassificationFamily.OTHER_ENSEMBLE,
    "Other_Methods_(OM)": ClassificationFamily.OTHER_METHOD,
    "Partial_Least_Squares_and_Principal_Component_Regression_(PLSR)": ClassificationFamily.PARTIAL_LEAST_SQUARES_AND_PRINCIPAL_COMPONENT_REGRESSION,
    "Random_Forests_(RF)": ClassificationFamily.RANDOM_FOREST,
    "Rule-Based_Methods_(RL)": ClassificationFamily.RULE_BASED_METHOD,
    "Stacking_(STC)": ClassificationFamily.STACKING,
    "Support_Vector_Machines_(SVM)": ClassificationFamily.SVM
}


class Task(Enum):
    MULTICLASS = auto()
    BINARY = auto()
    UNKNOWN = auto()


TASK_MAPPING = {"multiclass_classification": Task.MULTICLASS,
                "binary_classification": Task.BINARY}


class BaseExperiment(abc.ABC):
    def __init__(self):
        self.name = self.__class__.__name__


    @abc.abstractmethod
    def rank_families(self, dataset: pd.DataFrame, task: Task, seed, duration: int):
        pass


    def perform_experiments(self, seeds):
        results = defaultdict(dict)
        for seed in seeds:
            for duration in TEST_DURATIONS:
                result = self.__perform_experiment(seed, duration)
                results[seed][duration] = result
        self.__save_to_json(self.name, results)


    def __perform_experiment(self, seed, duration):
        experiment_results = dict()
        parameters = self._get_parameters()
        family_scores = self.__get_family_scores()
        for (_, dataset_parameter), (_, dataset_scores) in zip(parameters.iterrows(), family_scores.iterrows()):
            dataset_name = dataset_parameter["name"]
            num_classes = dataset_parameter["num_classes"]
            dataset_scores = {_FAMILY_MAPPING[family]: score for family, score in dict(dataset_scores[1:]).items()}
            max_score = max(dataset_scores.values())
            dataset = self.__get_dataset(dataset_name)
            task = Task.UNKNOWN
            if num_classes == 2:
                task = Task.BINARY
            elif 2 < num_classes < 50:
                # realistically it would be more of regression then
                task = Task.MULTICLASS
            if dataset is None:
                logging.warning(f"Dataset {dataset_name} not found")
                continue
            ranked_families = self.rank_families(dataset, task, seed, duration)
            scores_list = []
            current_max_score = 0
            for family in ranked_families:
                current_max_score = max(current_max_score, dataset_scores.get(family))
                scores_list.append(current_max_score / max_score)
            experiment_results[dataset_name] = scores_list
        return experiment_results
                

    def _get_parameters(self):
        return pd.read_csv("data/parameters.csv")

    def __get_family_scores(self):
        return pd.read_csv("data/family_scores.csv")
    
    def __get_dataset(self, name: str):
        csv_path = f"datasets/AutoIRAD-datasets/*{name}*.csv"
        dat_path = f"datasets/AutoIRAD-datasets/*{name}*.dat"
        found_csv = list(glob(csv_path))
        found_dat = list(glob(dat_path))
        if not found_csv and not found_dat:
            return None
        if len(found_csv) + len(found_dat) > 1:
            path = min(found_csv + found_dat, key=len)
            if "csv" in path:
                found_dat = []
                found_csv = [path]
            else:
                found_csv = []
                found_dat = [path]
        if found_csv:
            return pd.read_csv(found_csv[0])
        if found_dat:
            df = pd.read_table(found_dat[0])
            return df.iloc[:, 1:] # first column is index
                      

    def __save_to_json(self, library_name: str, data: dict) -> None:
        output_path = f"benchmark/outputs/{library_name}-{datetime.now()}.json"
        if not os.path.exists("benchmark/outputs"):
            os.makedirs("benchmark/outputs")
        with open(output_path, "w") as f:
            json.dump(data, f)
