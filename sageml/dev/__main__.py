""" CLI tool for developers """
import os
import pickle
import click
import pandas as pd
import torch

from sageml.dev.functions import (
    _extract,
    create_datasets,
    create_pydataset,
)
from sageml.workflow.algorithms_evaluations import evaluate_algorithm
from sageml.workflow.train_model import train_meta_model
from sageml.workflow.utils import read_data_file
PATH = os.path.join('sageml', 'meta_learning', 'meta-dataset')
MODEL_PATH = os.path.join('sageml', 'meta_learning', 'model')


@click.group()
def cli():
    """ Main CLI group """


@cli.command()
@click.option('--dataset_path', required=False, default=PATH, help='Path to the dataset the model will be trained on')
@click.option('--save', required=False, default=MODEL_PATH, help='Path that the model will be saved on.')
def train(dataset_path: str, save: str):
    """ Trains the model """
    score_dataset = pd.read_csv(os.path.join(dataset_path, 'scores.csv'), index_col=0)
    param_dataset = pd.read_csv(os.path.join(dataset_path, 'parameters.csv'), index_col=0)
    model, preprocessor, config = train_meta_model(score_dataset, param_dataset, 700)
    model = model.cpu()
    torch.save(model.state_dict(), os.path.join(save, 'model.pth'))
    with open(os.path.join(save, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(save, 'model_params.pkl'), 'wb') as f:
        pickle.dump(config, f)


@cli.group()
def dataset():
    """ Main meta-dataset related operations """


@dataset.command()
@click.argument('path', required=True, type=click.Path())
@click.option('--save', type=click.Path(), default=PATH, help='Where to save the file')
@click.option('--max_datasets', type=click.INT, help='Maximum number of datasets')
def create(path: str, save: str | None, max_datasets: int | None):
    """ Creates meta-dataset """
    click.echo('Creating new dataset!')
    score_dataset, param_dataset = create_datasets(path, -1 if max_datasets is None else max_datasets)
    click.echo('Dataset created successfully!')
    if save is not None:
        click.echo('Saving dataset ...')
        score_dataset.to_csv(os.path.join(save, 'scores.csv'))
        param_dataset.to_csv(os.path.join(save, 'parameters.csv'))
        click.echo('Dataset saved successfully')
    else:
        print(score_dataset)


@dataset.command()
@click.argument('path', required=True, type=click.Path())
@click.option('-r', is_flag=True, help='Interprets path as a directory of datasets')
@click.option('-p', is_flag=True, help='Interprets path as a name of preset dataset.')
def add(path: str, r: bool, p: bool):
    """ Adds entry to the dataset """
    score_dataset = pd.read_csv(os.path.join(PATH, 'scores.csv'))
    param_dataset = pd.read_csv(os.path.join(PATH, 'parameters.csv'))
    if p is True:
        if path == 'pydataset':
            new_score_dataset, new_param_dataset = create_pydataset()
            score_dataset = pd.concat([score_dataset, new_score_dataset])
            param_dataset = pd.concat([param_dataset, new_param_dataset])
    elif r is True:
        new_score_dataset, new_param_dataset = create_datasets(path)
        score_dataset = pd.concat([score_dataset, new_score_dataset])
        param_dataset = pd.concat([param_dataset, new_param_dataset])
    else:
        new_dataset = read_data_file(path)
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        score_dataset = pd.concat([score_dataset, evaluate_algorithm(new_dataset, dataset_name).to_frame().T], ignore_index=True)
        param_dataset = pd.concat([param_dataset, _extract(new_dataset, dataset_name).to_frame().T], ignore_index=True)
    score_dataset.to_csv(os.path.join(PATH, 'scores.csv'))
    param_dataset.to_csv(os.path.join(PATH, 'parameters.csv'))


if __name__ == '__main__':
    cli()
