from typing import List
import pandas as pd
from datasets import get_iris, get_wine, get_breast_cancer, get_digits, get_adult, get_tips, get_titanic
# get_heart_disease loads incorrectly
from quick_ai.algorithms import NeuralNetworkModel, XGBoostClassifier, sklearn_models
from quick_ai.preprocessing import Normalizer, NanImputer, OneHotEncoder, LabelEncoder
from quick_ai.forecast import HyperTuner, StatisticalParametersExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from pydataset import data


def generate_dataset(models, datasets):
    for dataset in datasets:
        data_train, data_test, target_train, target_test = train_test_split(
            *dataset, test_size=0.2)
        for preprocessor in [NanImputer, Normalizer]:
            preprocessor_instance = preprocessor()
            data_train = preprocessor_instance.fit_transform(data_train)
            target_train = preprocessor_instance.fit_transform_target(
                target_train)
            data_test = preprocessor_instance.transform(data_test)
            target_test = preprocessor_instance.transform_target(target_test)
        ohe = OneHotEncoder()
        data_train = ohe.fit_transform(data_train)
        data_test = ohe.transform(data_test)
        le = LabelEncoder()
        target_train = le.fit_transform_target(target_train)
        target_test = le.transform_target(target_test)
        extractor = StatisticalParametersExtractor(
            data_train, target_train)
        description = extractor.describe_dataset()
        scores = {}
        for model in models:
            print(model.__name__)
            tuner = HyperTuner()
            try:
                params = tuner.optimize_hyperparameters(model, (data_train, target_train), description.task,
                                                        description.num_classes, description.target_features, device='cuda', trials=10)
            except Exception as e:
                print(e)
                params = {}
            try:
                model_instance = model(**params)
                model_instance.train(data_train, target_train)
            except Exception as e:
                print(0)
                print(e)
                scores[model.__name__] = 0
                continue
            try:
                result = sum(model_instance.predict(data_test) ==
                             target_test)/len(target_test)
                print(result)
                scores[model.__name__] = result
            except Exception as e:
                print(e)
                print(0)
                scores[model.__name__] = 0
        record = {**description.dict(), **scores}
        df = pd.DataFrame([record])
        score_columns = [model.__name__ for model in models]
        df[score_columns] = minmax_scale(df[score_columns], axis=1)
        df.to_csv("results.csv", mode='a', header=False, index=False)


ALL_MODELS = [NeuralNetworkModel, XGBoostClassifier] + \
    list(sklearn_models.values())

PY_DATASETS = [data(id) for id in data()['dataset_id']]


def adapt_pydatasets(datasets: List[pd.DataFrame]):
    for dataset in datasets:
        if 'object' in dataset.dtypes.to_numpy():
            categorical_column = dataset.select_dtypes(
                include='object').iloc[:, 0]
            dataset.drop(categorical_column.name, axis=1, inplace=True)
            yield dataset, categorical_column


ALL_DATASETS = list(adapt_pydatasets(PY_DATASETS))
generate_dataset(ALL_MODELS, ALL_DATASETS)
