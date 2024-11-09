import os
from typing import Tuple
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw
from turbo_ml.preprocessing import Normalizer


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

        for j in range(1):
            tsne = TSNE(n_components=2, random_state=j, n_jobs=-1, perplexity=20)
            X_embedded = tsne.fit_transform(X)
            X_normalized = Normalizer().fit_transform(
                pd.DataFrame(X_embedded, columns=['x', 'y']))
            X_normalized['x'] *= (resolution[0]-1)
            X_normalized['y'] *= (resolution[1]-1)
            X_normalized['class'] = y

            unique_classes = y.unique()
            num_classes = len(unique_classes)
            image = Image.new("RGB", resolution, (255, 255, 255))
            draw = ImageDraw.Draw(image)
            colormap = plt.cm.get_cmap("tab10", num_classes)
            class_colors = {cls: tuple(
                int(c * 255) for c in colormap(i)[:3]) for i, cls in enumerate(unique_classes)}
            for _, row in X_normalized.iterrows():
                width, height, cls = int(row['x']), int(row['y']), row['class']
                color = class_colors.get(cls, (0, 0, 0))
                draw.point((width, height), fill=color)
            image.save(images_dir + '/' + str(i)
                       # + '_' + str(j)
                       + '.png')


if __name__ == '__main__':
    generate_AutoIRAD_dataset(results_path=os.path.join('data', 'family_scores.csv'), datasets_dir=os.path.join(
        'datasets', 'AutoIRAD-datasets'), path1='data/best_family.csv', images_dir=os.path.join('autoIRAD', 'images'))
