import os
import pprint
import pandas as pd
from turbo_ml.preprocessing import Normalizer, NanImputer, Encoder, LabelEncoder
from turbo_ml.meta_learning.dataset_parameters import (
    SimpleMetaFeatures,
    CombinedMetaFeatures,
    StatisticalMetaFeatures,
    PCAMetaFeatures,
    RipserFeatures,
    BallMapperFeatures,
)
from turbo_ml.workflow.utils import read_data_file, list_dataset_files
from tqdm import tqdm
from prefect import flow
from pydataset import data


@flow(name="Generate Training Parameters")
def generate_training_parameters(
    datasets_dir: str = os.path.join("datasets", "AutoIRAD-datasets"),
    output_path="parameters.csv",
    meta_data_extractor=SimpleMetaFeatures(),
    preprocessors=[NanImputer, Normalizer, Encoder],
):
    def measure(X, y):
        return meta_data_extractor(X, y, as_dict=True)

    names = list_dataset_files(datasets_dir)

    dataframe = None
    parameters = {}
    for dataset_name, path in tqdm(names, total=len(names)):
        try:
            dataset = read_data_file(path)
            X = dataset.drop(dataset.columns[-1], axis=1)
            y = dataset[dataset.columns[-1]]

            for preprocessor in preprocessors:
                X = preprocessor().fit_transform(X)
                y = preprocessor().fit_transform_target(y)

            parameters = measure(X, y)

            if dataframe is None:
                dataframe = pd.DataFrame([parameters])
                dataframe.insert(0, "name", dataset_name)
                continue

            parameters["name"] = dataset_name
            dataframe = pd.concat([dataframe, pd.DataFrame([parameters])], axis=0)

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    if dataframe is not None and output_path is not None:
        dataframe.to_csv(output_path, index=False)
    return dataframe


@flow(name="pydataset parameters")
def generate_training_parameters_pydataset(
    output_path="parameters.csv",
    meta_data_extractor=SimpleMetaFeatures(),
    preprocessors=[NanImputer, Normalizer, Encoder],
):
    def measure(X, y):
        return meta_data_extractor(X, y, as_dict=True)

    names = data()["dataset_id"]

    dataframe = None
    parameters = {}
    for dataset_name in tqdm(names, total=len(names)):
        try:
            dataset = data(dataset_name)
            X = dataset.drop(dataset.columns[-1], axis=1)
            y = dataset[dataset.columns[-1]]

            for preprocessor in preprocessors:
                X = preprocessor().fit_transform(X)
                y = preprocessor().fit_transform_target(y)

            parameters = measure(X, y)

            if dataframe is None:
                dataframe = pd.DataFrame([parameters])
                dataframe.insert(0, "name", dataset_name)
                continue

            parameters["name"] = dataset_name
            dataframe = pd.concat([dataframe, pd.DataFrame([parameters])], axis=0)

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    if dataframe is not None and output_path is not None:
        dataframe.to_csv(output_path, index=False)
    return dataframe


if __name__ == "__main__":
    generate_training_parameters_pydataset(meta_data_extractor=BallMapperFeatures())
