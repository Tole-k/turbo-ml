import os
from typing import Tuple
import pandas as pd

def read_data_file(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        with open(path, 'r') as f:
            return pd.read_csv(f)
    if path.endswith('.dat'):
        with open(path, 'r') as f:
            return pd.read_csv(f, delimiter='\t').drop('Unnamed: 0', axis=1)
    raise ValueError(f'File format not supported for file: {path}')

def list_dataset_files(datasets_dir: str) -> list[Tuple[str, str]]:
    def is_dataset_file(name_path: Tuple[str, str]) -> bool:
        _, path = name_path
        return path.endswith('.csv') or path.endswith('.dat')
    names = []
    directory = os.path.join(datasets_dir)
    for root, dirs, files in os.walk(directory):
        names.extend([(file.rsplit('.', 1)[0], os.path.join(root, file)) for file in files])

    names = list(filter(is_dataset_file, names))
    names.sort()
    return names
