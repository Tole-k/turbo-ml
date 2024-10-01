import pandas as pd
from datasets import get_iris, get_wine, get_breast_cancer, get_digits, get_adult, get_tips, get_titanic
# get_heart_disease loads incorrectly
from quick_ai.algorithms import NeuralNetworkModel, XGBoostClassifier, sklearn_models
from quick_ai.preprocessing import Normalizer, NanImputer, OneHotEncoder, LabelEncoder
from quick_ai.forecast import HyperTuner, StatisticalParametersExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def generate_dataset(models, datasets):
    records = []
    for dataset in datasets:
        print(dataset.__name__)
        data_train, data_test, target_train, target_test = train_test_split(
            *dataset(), test_size=0.2)
        for preprocessor in [NanImputer, Normalizer]:
            print(data_train)
            print(target_train)
            preprocessor_instance = preprocessor()
            data_train = preprocessor_instance.fit_transform(data_train)
            target_train = preprocessor_instance.fit_transform_target(
                target_train)
            data_test = preprocessor_instance.transform(data_test)
            target_test = preprocessor_instance.transform_target(target_test)
        print(data_train)
        print(target_train)
        ohe = OneHotEncoder()
        data_train = ohe.fit_transform(data_train)
        data_test = ohe.transform(data_test)
        le = LabelEncoder()
        target_train = le.fit_transform_target(target_train)
        target_test = le.transform_target(target_test)
        extractor = StatisticalParametersExtractor(
            data_train, target_train)
        description = extractor.describe_dataset()
        print(description)
        scores = {}
        for model in models:
            print(model.__name__)
            tuner = HyperTuner()
            try:
                params = tuner.optimize_hyperparameters(model, (data_train, target_train), description.task,
                                                        description.num_classes, description.target_features, device='cuda', trials=10)
            except Exception:
                params = {}
            model_instance = model(**params)
            model_instance.train(data_train, target_train)
            try:
                result = sum(model_instance.predict(data_test) ==
                             target_test)/len(target_test)
                print(result)
                scores[model.__name__] = result
            except Exception:
                print(0)
                scores[model.__name__] = 0
        records.append(description.dict() | scores)
    df = pd.DataFrame.from_records(records)
    score_columns = [model.__name__ for model in models]
    df[score_columns] = minmax_scale(df[score_columns], axis=1)
    df.to_csv("results.csv")
    return df


ALL_MODELS = [NeuralNetworkModel, XGBoostClassifier] + \
    list(sklearn_models.values())

ALL_DATASETS = [get_iris, get_wine, get_breast_cancer,
                get_digits, get_titanic]

generate_dataset(ALL_MODELS, ALL_DATASETS)
