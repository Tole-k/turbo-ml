import os
from typing import Tuple
import pandas as pd
from turbo_ml.preprocessing import Normalizer, NanImputer, OneHotEncoder, LabelEncoder
from turbo_ml.meta_learning.dataset_parameters import SimpleMetaFeatures, CombinedMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from turbo_ml.meta_learning.dataset_parameters.topological import RipserFeatures, BallMapperFeatures
from tqdm import tqdm
from prefect import flow

@flow(name='Generate Training Parameters')
def generate_training_parameters(datasets_dir: str = os.path.join('datasets', 'AutoIRAD-datasets'),
                                 output_path='parameters.csv', meta_data_extractor=SimpleMetaFeatures(),
                                 preprocessor=[NanImputer, Normalizer, OneHotEncoder]):

    def read_csv(path:str) -> pd.DataFrame:
        if path.endswith('.csv'):
            with open(path, 'r') as f:
                return pd.read_csv(f)
        if path.endswith('.dat'):
            with open(path, 'r') as f:
                return pd.read_csv(f, delimiter='\t').drop('Unnamed: 0', axis=1)
        raise ValueError(f'File format not supported for file: {path}')    

    def measure(X, y):
        return meta_data_extractor(X, y, as_dict=True)

    def is_dataset_file(name_path:Tuple[str, str]) -> bool:
        _, path = name_path
        return path.endswith('.csv') or path.endswith('.dat')

    names = []
    directory = os.path.join(datasets_dir)
    for root, dirs, files in os.walk(directory):
        names.extend([(file, os.path.join(root, file)) for file in files])

    names = list(filter(is_dataset_file, names))        
    names.sort()

    print('Datasets:', names)
    dataframe = None
    parameters = {}
    for i, (dataset_name, path) in tqdm(enumerate(names), total=len(names)):
        try:
            dataset = read_csv(path)
            X = dataset.drop(dataset.columns[-1], axis=1)
            y = dataset[dataset.columns[-1]]

            for preprocessor in [NanImputer, Normalizer, OneHotEncoder]:
                X = preprocessor().fit_transform(X)
                y = preprocessor().fit_transform_target(y)

            parameters = measure(X, y)
            
            if dataframe is None:
                dataframe = pd.DataFrame([parameters])
                dataframe.insert(0, 'name', dataset_name)
                continue
            
            parameters['name'] = dataset_name
            dataframe = pd.concat([dataframe, pd.DataFrame([parameters])], axis=0)
            
        except Exception as e:
            print(f'Error processing dataset {dataset_name}: {e}')
            continue
    if dataframe is not None and output_path is not None:
        dataframe.to_csv(output_path, index=False)
    return dataframe


if __name__ == '__main__':
    print(generate_training_parameters(output_path='ball_mapper_scores.csv'))
