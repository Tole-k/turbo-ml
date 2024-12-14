from typing import List
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from .base import MetaFeature
import numpy as np
from datasets import get_iris
from pyballmapper import BallMapper
from ripser import ripser
import pandas as pd
from decorify import mute

class BallMapperFeatures(MetaFeature):
    def __init__(self, epsilons:List[int] = [0.25, 1, 5, 25], verbose:bool=False):
        self.epsilons = epsilons
        self.verbose = verbose

    @mute('warning')
    def __call__(self, dataset, target_data, as_dict=False):
        self.epsilons = [0.25, 1, 5, 25]
        features = {}
        for eps in self.epsilons:
            bm = BallMapper(X=dataset.values, eps=eps, coloring_df=pd.DataFrame(
                target_data, columns=['target']))

            if self.verbose:
                if len(bm.points_covered_by_landmarks) == 0:
                    print(f'No landmarks found for eps={eps}')
                    continue
                my_red_palette = cm.get_cmap("viridis")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax = bm.draw_networkx(coloring_variable='target',
                                    ax=ax, color_palette=my_red_palette)
                plt.show()

            points = bm.points_covered_by_landmarks.values()
            points = points if len(points) > 0 else [0]
            point_lengths = np.array(list(map(len, bm.points_covered_by_landmarks.values())))
            features[f'number_of_landmarks_at_{eps}'] = point_lengths.shape[0]
            features[f'mean_len_at_{eps}'] = point_lengths.mean()
            features[f'min_len_at_{eps}'] = point_lengths.min()
            features[f'max_len_at_{eps}'] = point_lengths.max()
            features[f'std_len_at_{eps}'] = point_lengths.std()
            features[f'median_len_at_{eps}'] = np.median(point_lengths)

        if as_dict:
            return features
        return np.array(list(features.values()))

class RipserFeatures(MetaFeature):
    def __call__(self, dataframe, target_data, as_dict=False):
        diagrams = ripser(dataframe.values, maxdim=2)['dgms']

        features = {}
        betti_numbers = []
        # plot_diagrams(diagrams, show=True)
        for dim, diagram in enumerate(diagrams):
            betti_numbers.append(len(diagram))

            finite_lifespans = diagram[np.isfinite(diagram[:, 1])]
            lifespans = finite_lifespans[:, 1] - finite_lifespans[:, 0]

            total_pers = np.sum(lifespans)
            pers_entropy = -np.sum((lifespans / total_pers)
                                   * np.log(lifespans / total_pers))

            if len(lifespans) == 0:
                lifespans = [0]
            features[f'dim_{dim}_total_persistence'] = total_pers
            features[f'dim_{dim}_max_persistence'] = np.max(lifespans)
            features[f'dim_{dim}_mean_persistence'] = np.mean(lifespans)
            features[f'dim_{dim}_median_persistence'] = np.median(lifespans)
            features[f'dim_{dim}_std_persistence'] = np.std(lifespans)
            features[f'dim_{dim}_persistence_entropy'] = pers_entropy

        features['euler_characteristic'] = sum(
            (-1)**i * b for i, b in enumerate(betti_numbers))

        for dim, betti in enumerate(betti_numbers):
            features[f'betti_{dim}'] = betti
        if as_dict:
            return features
        return np.array(list(features.values()))


if __name__ == '__main__':
    # import pprint
    # dataset, target = get_iris()
    # parameters = RipserFeatures()(dataset, target, as_dict=dict)
    # pprint.pprint(parameters)
    dataset, target = get_iris()
    parameters = BallMapperFeatures()(dataset, target)
    print(parameters)
    # parameters = BallMapperFeatures()(dataset, target, as_dict=True)
    # print(parameters)
