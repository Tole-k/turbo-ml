from typing import List
import pandas as pd
from datasets import get_iris, get_wine, get_breast_cancer, get_digits, get_adult, get_tips, get_titanic
# get_heart_disease loads incorrectly
from turbo_ml.algorithms import NeuralNetworkModel, XGBoostClassifier, sklearn_models
from turbo_ml.preprocessing import Normalizer, NanImputer, OneHotEncoder, LabelEncoder
from turbo_ml.forecast import HyperTuner, StatisticalParametersExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from pydataset import data

from turbo_ml.utils import options


def generate_dataset(models, datasets, optuna_trials=10, device='cpu', threads=1, path='results.csv'):
    for dataset in datasets:
        if isinstance(dataset, tuple):
            name = dataset[1]
            dataset = dataset[0]
        if callable(dataset):
            name = dataset.__name__
            dataset = dataset()
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
                                                        description.num_classes, description.target_features, device, optuna_trials, threads)
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
        description_dict = description.dict()
        description_dict['name'] = name
        record = {**description_dict, **scores}
        df = pd.DataFrame([record])
        names = df['name']
        df.drop('name', axis=1, inplace=True)
        df.insert(0, 'name', names)
        score_columns = [model.__name__ for model in models]
        df[score_columns] = minmax_scale(df[score_columns], axis=1)
        df.to_csv(path, mode='a', header=False, index=False)


ALL_MODELS = [NeuralNetworkModel, XGBoostClassifier] + \
    list(sklearn_models.values())

PY_DATASETS = [(data(id), name) for id, name in zip(
    data()['dataset_id'], data()['title'])]


def adapt_pydatasets(datasets: List[pd.DataFrame]):
    for dataset in datasets:
        name = dataset[1]
        dataset = dataset[0]
        if 'object' in dataset.dtypes.to_numpy():
            categorical_column = dataset.select_dtypes(
                include='object').iloc[:, -1]
            dataset.drop(categorical_column.name, axis=1, inplace=True)
            yield ((dataset, categorical_column), name)


ALL_DATASETS = [get_iris, get_wine, get_breast_cancer, get_digits, get_adult,
                get_tips, get_titanic] + list(adapt_pydatasets(PY_DATASETS))

if __name__ == '__main__':
    generate_dataset(ALL_MODELS, ALL_DATASETS, optuna_trials=10,
                     device=options.device, threads=options.threads, path='results2.csv')
