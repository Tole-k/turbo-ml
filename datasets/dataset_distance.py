import timeit
from typing import Tuple, List
from sklearn.decomposition import PCA
import numpy as np

from datasets import get_AutoIRAD_datasets
from datasets.otdd import ArrayDataset, POTDistance, EarthMoversCost
import pandas as pd
from sklearn.preprocessing import Normalizer
from turbo_ml.preprocessing import OneHotEncoder, NanImputer


def process_datasets(datasets, names):
    datas = []
    nams = []
    for dataset, name in zip(datasets, names):
        data = dataset.drop(dataset.columns[-1], axis=1)
        data = NanImputer().fit_transform(data)
        data = OneHotEncoder().fit_transform(data)
        target = dataset.iloc[:, -1]
        datas.append(
            (data, target.astype(str)))
        nams.append(name)
    datasets = datas.copy()
    names = nams.copy()
    return datasets, names


def calculate_distance(dataset0: Tuple[pd.DataFrame, pd.DataFrame], dataset1: Tuple[pd.DataFrame, pd.DataFrame]):
    dataset0_x, dataset0_y = dataset0
    dataset1_x, dataset1_y = dataset1
    max_width = min(dataset0_x.shape[0], dataset0_x.shape[1],
                    dataset1_x.shape[0], dataset1_x.shape[1])
    pca = PCA(n_components=max_width)
    dataset0_x = pca.fit_transform(dataset0_x).astype(np.float64)
    dataset1_x = pca.fit_transform(dataset1_x).astype(np.float64)

    combined = np.vstack((dataset0_x, dataset1_x))
    normalizer = Normalizer().fit(combined)

    dataset0_x = normalizer.transform(dataset0_x)
    dataset1_x = normalizer.transform(dataset1_x)

    ad1 = ArrayDataset(dataset0_x, dataset0_y)
    ad2 = ArrayDataset(dataset1_x, dataset1_y)

    distance_function = POTDistance(distance_metric='euclidean')
    cost_function = EarthMoversCost(distance_function)

    cost, coupling, OTDD_matrix, class_distances, class_x_dict, class_y_dict = cost_function.distance_with_labels(
        ad1.features, ad2.features, ad1.labels, ad2.labels)
    return cost


def calculate_distances(datasets: List[pd.DataFrame], names: List[str]):
    scores = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        start = timeit.default_timer()
        for j in range(i + 1, len(datasets)):
            dataset0 = datasets[i]
            dataset1 = datasets[j]

            cost = calculate_distance(
                dataset0, dataset1)
            print(f'{i}. {names[i]} vs {j}, {names[j]}: {cost}')
            scores[i, j] = cost
            scores[j, i] = cost
        stop = timeit.default_timer()
        print(f'Time: {stop - start} seconds')
    df = pd.DataFrame(scores, columns=names, index=names)
    return df


if __name__ == '__main__':
    datasets, names = process_datasets(*get_AutoIRAD_datasets())
    df = calculate_distances(datasets, names)
    df.to_csv('AutoIRAD-dataset-distances-ohe.csv')
