""" Workflow utils """
import os
import pandas as pd


def read_data_file(path: str) -> pd.DataFrame:
    """Reads given dataset.

    Args:
        path (str): Path to the file.

    Raises:
        ValueError: If file format is not supported

    Returns:
        pd.DataFrame: dataset
    """
    if path.endswith('.csv'):
        with open(path, 'r', encoding='utf8') as f:
            dataset = pd.read_csv(f)
    elif path.endswith('.dat'):
        with open(path, 'r', encoding='utf8') as f:
            dataset = pd.read_csv(f, delimiter='\t').drop('Unnamed: 0', axis=1)
    else:
        raise ValueError(f'File format not supported for file: {path}')
    return dataset


def list_dataset_files(datasets_dir: str) -> list[str]:
    """Lists all datasets in directory.

    Args:
        datasets_dir (str): Directory with datasets.

    Returns:
        list[tuple[str, str]]: List of ..., ... pairs.
    """
    def is_dataset_file(path: str) -> bool:
        return path.endswith('.csv') or path.endswith('.dat')
    names = []
    directory = os.path.join(datasets_dir)
    for root, _, files in os.walk(directory):
        for file in files:
            if not is_dataset_file(file):
                continue
            names.append(os.path.join(root, file))

    names.sort()
    return names
