from sklearn import datasets
import pandas as pd
import certifi
import ssl
import requests
from io import StringIO


def fetching():
    ssl._create_default_https_context = ssl.create_default_context(
        cafile=certifi.where())


def get_iris() -> tuple[pd.DataFrame, pd.Series]:
    bunch = datasets.load_iris(as_frame=True)
    return bunch['data'], bunch['target']


def get_wine() -> tuple[pd.DataFrame, pd.Series]:
    bunch = datasets.load_wine(as_frame=True)
    return bunch['data'], bunch['target']


def get_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    bunch = datasets.load_breast_cancer(as_frame=True)
    return bunch['data'], bunch['target']


def get_digits() -> tuple[pd.DataFrame, pd.Series]:
    bunch = datasets.load_digits(as_frame=True)
    return bunch['data'], bunch['target']


def get_diabetes() -> tuple[pd.DataFrame, pd.Series]:
    bunch = datasets.load_diabetes(as_frame=True)
    return bunch['data'], bunch['target']


def get_linnerud() -> tuple[pd.DataFrame, pd.DataFrame]:
    bunch = datasets.load_linnerud(as_frame=True)
    return bunch['data'], bunch['target']


def get_titanic() -> tuple[pd.DataFrame, pd.Series]:
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
    response = requests.get(url)
    titanic = pd.read_csv(StringIO(response.text))
    target = pd.Series(titanic['alive'])
    titanic.drop(columns=['alive'], inplace=True)
    titanic.drop(columns=['survived'], inplace=True)
    return titanic, target


def get_tips() -> tuple[pd.DataFrame, pd.Series]:
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
    response = requests.get(url)
    tips = pd.read_csv(StringIO(response.text))
    target = pd.Series(tips['tip'])
    tips.drop(columns=['tip'], inplace=True)
    return tips, target


def get_adult() -> tuple[pd.DataFrame, pd.Series]:
    url = 'https://www.openml.org/data/get_csv/1595261/adult-census.arff'
    response = requests.get(url)
    adult = pd.read_csv(StringIO(response.text))
    target = pd.Series(adult['class'])
    adult.drop(columns=['class'], inplace=True)
    adult['age'] = pd.cut(adult['age'], bins=3, labels=[
                          'young', 'adult', 'old'])
    adult['capital-gain'] = pd.cut(adult['capital-gain'],
                                   bins=3, labels=['low', 'medium', 'high'])
    adult['capital-loss'] = pd.cut(adult['capital-loss'],
                                   bins=3, labels=['low', 'medium', 'high'])
    adult['hours-per-week'] = pd.cut(adult['hours-per-week'],
                                     bins=3, labels=['low', 'medium', 'high'])
    adult['fnlwgt'] = pd.cut(adult['fnlwgt'], bins=3, labels=[
                             'low', 'medium', 'high'])
    adult['education-num'] = pd.cut(adult['education-num'],
                                    bins=3, labels=['low', 'medium', 'high'])
    return adult, target


def get_heart_disease() -> tuple[pd.DataFrame, pd.Series]:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    response = requests.get(url)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    heart_disease = pd.read_csv(StringIO(response.text),
                                header=None, names=columns)
    target = pd.Series(heart_disease['num'])
    heart_disease.drop(columns=['num'], inplace=True)
    return heart_disease, target
