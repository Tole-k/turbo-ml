import os
import abc
import logging
from glob import glob
from enum import Enum, auto
from typing import List
from datetime import datetime

import pandas as pd

SEEDS = [0, 2, 3, 4, 5]
TEST_DURATIONS = [180, 900]  # in seconds


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


class YamlWriter:
    def __init__(self, library_name):
        self.path: str = f"benchmark/outputs/{
            library_name}-{datetime.now()}.yaml"
        self.file = None
        self.indent_string = "    "
        self.indent_size = 0

    def __enter__(self):
        if not os.path.exists("benchmark/outputs"):
            os.makedirs("benchmark/outputs")
        self.file = open(self.path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def increase_indent(self, name: str):
        self.file.write(self.indent_string * self.indent_size + f"{name}:\n")
        self.indent_size += 1

    def decrease_indent(self):
        self.indent_size -= 1

    def change_indent_name(self, name: str):
        self.decrease_indent()
        self.increase_indent(name)

    def add_partial_result(self, dataset_name: str, result: List[str]):
        prefix = self.indent_string * self.indent_size + "- "
        self.file.write(prefix + f"{dataset_name}: {result}\n")


class BaseExperiment(abc.ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def rank_families(self, dataset: pd.DataFrame, dataset_name, task: Task, seed, duration: int) -> List[ClassificationFamily]:
        pass

    def perform_experiments(self, seeds=SEEDS, durations=TEST_DURATIONS):
        with YamlWriter(self.name) as writer:
            self.writer = writer
            for seed in seeds:
                writer.increase_indent(seed)
                for duration in durations:
                    writer.increase_indent(duration)
                    self._perform_experiment(seed, duration)
                    writer.decrease_indent()
                writer.decrease_indent()

    def _perform_experiment(self, seed, duration):
        parameters = self._get_parameters()
        for (_, dataset_parameter) in parameters.iterrows():
            dataset_name = dataset_parameter["name"]
            num_classes = dataset_parameter["num_classes"]
            dataset = self.__get_dataset(dataset_name)
            task = Task.UNKNOWN
            if num_classes == 2:
                task = Task.BINARY
            elif 2 < num_classes < 10000000:
                # realistically it would be more of regression then
                task = Task.MULTICLASS
            if dataset is None:
                logging.warning(f"Dataset {dataset_name} not found")
                continue
            ranked_families = self.rank_families(
                dataset, dataset_name, task, seed, duration)
            if ranked_families is None:
                self.writer.add_partial_result(dataset_name, [])
                continue
            ranked_families_names = []
            for family in ranked_families:
                if family.name not in ranked_families_names:
                    ranked_families_names.append(family.name)
            self.writer.add_partial_result(dataset_name, ranked_families_names)

    def _get_parameters(self):
        return pd.read_csv("data/parameters.csv")

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
            return df.iloc[:, 1:]  # first column is index
