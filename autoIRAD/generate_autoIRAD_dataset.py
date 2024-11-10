import os
from typing import Tuple
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw
from turbo_ml.preprocessing import Normalizer
import matplotlib
matplotlib.use('Agg')


def generate_AutoIRAD_dataset(results_path: str = os.path.join('datasets', 'results_algorithms.csv'), datasets_dir: str = os.path.join('datasets', 'AutoIRAD-datasets'), path1='scores.csv', images_dir: str = os.path.join('autoIRAD', 'images'), resolution: Tuple[int, int] = (100, 100)):
    with open(results_path, 'r') as f:
        scores = pd.read_csv(f, index_col=0)
    sets = {}
    names = []
    directory = os.path.join(datasets_dir)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                with open(os.path.join(root, file), 'r') as f:
                    file = file.replace('.csv', '')
                    sets[file] = pd.read_csv(f)
                    names.append(file)
            if file.endswith('.dat'):
                with open(os.path.join(root, file), 'r') as f:
                    file = file.replace('_R', '')
                    file = file.replace('.dat', '')
                    sets[file] = pd.read_csv(
                        f, delimiter='\t').drop('Unnamed: 0', axis=1)
                    names.append(file)
    names.sort()
    for i, dataset_name in enumerate(names):
        dataset = sets[dataset_name]
        if dataset_name in scores.index:
            datasets_scores = scores.loc[dataset_name].to_dict()
        else:
            continue
        scores_array = np.array(list(datasets_scores.values()))
        best_family = np.argmax(scores_array)
        scores_df = pd.DataFrame(
            {'dataset': i, 'family': best_family}, index=[0])
        header = i == 0
        mode = 'w' if header else 'a'
        scores_df.to_csv(path1, mode=mode, header=header, index=False)

        X = dataset.drop(dataset.columns[-1], axis=1)
        y = dataset[dataset.columns[-1]]

        for j in range(10):
            print(dataset_name, j)
            tsne = TSNE(n_components=2,
                        n_jobs=-1, perplexity=20, random_state=j, init='random')
            X_embedded = tsne.fit_transform(X)
            X_normalized = Normalizer().fit_transform(
                pd.DataFrame(X_embedded, columns=['x', 'y']))

            X_normalized['x'] *= (resolution[0]-1)
            X_normalized['y'] *= (resolution[1]-1)
            X_normalized['x'] = X_normalized['x'].astype(int)
            X_normalized['y'] = X_normalized['y'].astype(int)
            X_normalized['class'] = y

            unique_classes = y.unique()

            plt.figure(
                figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
            for cls in unique_classes:
                class_data = X_normalized[X_normalized['class'] == cls]
                plt.scatter(class_data['x'], class_data['y'], label=str(cls))

            plt.axis('off')
            plt.savefig(images_dir + '/' + str(i) + '_' + dataset_name +
                        '_' + str(j) + '.png', dpi=100)
            plt.close()


if __name__ == '__main__':
    generate_AutoIRAD_dataset(results_path=os.path.join('data', 'family_scores.csv'), datasets_dir=os.path.join(
        'datasets', 'AutoIRAD-datasets'), path1='data/best_family.csv', images_dir=os.path.join('autoIRAD', 'images'), resolution=(512, 512))
