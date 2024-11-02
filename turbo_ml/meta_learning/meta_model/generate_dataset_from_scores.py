import os
import pandas as pd
# get_heart_disease loads incorrectly
from turbo_ml.preprocessing import Normalizer, NanImputer, OneHotEncoder, LabelEncoder
from turbo_ml.meta_learning.dataset_parameters import SimpleMetaFeatures, CombinedMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from turbo_ml.utils import options


def generate_dataset_from_scores(results_path: str = os.path.join('datasets', 'results_algorithms.csv'), datasets_dir: str = os.path.join('datasets', 'AutoIRAD-datasets'), path1='scores.csv', path2='parameters.csv'):
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
    parameters = {}
    for i, dataset_name in enumerate(names):
        dataset = sets[dataset_name]
        if dataset_name in scores.index:
            datasets_scores = scores.loc[dataset_name].to_dict()
        else:
            continue
        meta_features = [SimpleMetaFeatures(
        ), StatisticalMetaFeatures(), PCAMetaFeatures()]
        X = dataset.drop(dataset.columns[-1], axis=1)
        y = dataset[dataset.columns[-1]]
        # for preprocessor in [NanImputer, Normalizer]:
        #     X = preprocessor().fit_transform(X)
        #     y = preprocessor().fit_transform_target(y)
        # X = OneHotEncoder().fit_transform(X)
        # y = LabelEncoder().fit_transform_target(y)
        parameters = CombinedMetaFeatures(meta_features)(X, y, as_dict=True)
        scores_df = pd.DataFrame([datasets_scores])
        scores_df.insert(0, 'name', dataset_name)
        params_df = pd.DataFrame([parameters])
        params_df.insert(0, 'name', dataset_name)
        header = i == 0
        scores_df.to_csv(path1, mode='a', header=header, index=False)
        params_df.to_csv(path2, mode='a', header=header, index=False)


if __name__ == '__main__':
    generate_dataset_from_scores(results_path=os.path.join('datasets', 'results_algorithms.csv'), datasets_dir=os.path.join(
        'datasets', 'AutoIRAD-datasets'), path1='scores.csv', path2='parameters.csv')
