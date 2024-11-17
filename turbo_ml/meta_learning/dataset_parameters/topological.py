from matplotlib import colormaps as cm
from .base import MetaFeature
import numpy as np
from datasets import get_iris, get_adult
from pyballmapper import BallMapper

import pandas as pd


class BallMapperFeatures(MetaFeature):
    def __init__(self, num_intervals=10, num_balls=10):
        self.num_intervals = num_intervals
        self.num_balls = num_balls

    def __call__(self, dataset, target_data, as_dict=False):
        bm = BallMapper(X=dataset.values, eps=0.25, coloring_df=pd.DataFrame(
            target_data, columns=['target']))
        import matplotlib.pyplot as plt

        my_red_palette = cm.get_cmap("viridis")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = bm.draw_networkx(coloring_variable='target',
                              ax=ax, color_palette=my_red_palette)
        plt.show()
        return None


if __name__ == '__main__':
    dataset, target = get_iris()
    parameters = BallMapperFeatures()(dataset, target)

    # dataset, target = get_adult()
    # parameters = BallMapperFeatures()(dataset, target)
    # print(parameters)
    # parameters = BallMapperFeatures()(dataset, target, as_dict=True)
    # print(parameters)
