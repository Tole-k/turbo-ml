import os
from turbo_ml.base import __ALL_MODELS__
from PIL import Image
import numpy as np
import pandas as pd
from autoIRAD import AutoIRAD


from .utils import BaseExperiment, _FAMILY_MAPPING

import sys
sys.path.append('.')


class AutoIRADExperiment(BaseExperiment):
    def __init__(self, img_size):
        self.name = self.__class__.__name__
        self.img_size = img_size

        df = pd.read_csv('data/best_family.csv')
        df.set_index('dataset', inplace=True)

        family_scores = pd.read_csv('data/family_rma.csv', index_col=0)

        self.families = family_scores.columns

        with open(os.path.join('data', 'family_rma.csv'), 'r') as f:
            self.scores = pd.read_csv(f, index_col=0)

        images_dir = os.path.join('autoIRAD', 'images')
        self.images = []
        ys = []
        self.dataset_names = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.endswith(".png"):
                    with open(os.path.join(root, file), 'rb') as f:
                        image = Image.open(f)
                        image = np.asarray(image)
                        self.images.append(image)
                    y = df.loc[int(file.split('_')[0]), 'family']
                    ys.append(y)
                    self.dataset_names.append('_'.join(file.split('_')[1:-1]))
        self.images = np.array(self.images)
        self.images = np.resize(self.images, (len(self.images), *img_size, 3))
        self.ys = np.array(ys)

    def rank_families(self, dataset, dataset_name, *_):
        dataset_ids = [i for i, name in enumerate(
            self.dataset_names) if name == dataset_name]
        dataset_image = self.images[dataset_ids[0]]
        images_dropped = np.delete(self.images, dataset_ids, axis=0)
        ys_dropped = np.delete(self.ys, dataset_ids, axis=0)

        auto = AutoIRAD(resolution=self.img_size)
        auto.train(images_dropped, ys_dropped)
        family_nums = auto.predict(dataset_image[np.newaxis, :])
        families = self.families[family_nums]

        return [_FAMILY_MAPPING[family] for family in families]


if __name__ == "__main__":
    experiment = AutoIRADExperiment(img_size=(256, 256))
    experiment.perform_experiments(durations=[60])
