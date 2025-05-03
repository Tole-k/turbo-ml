""" CLI tool for developers """
import os
import click
import pandas as pd

from sageml.dev.functions import (
    create_datasets,
)
from sageml.workflow.train_model import train_meta_model
PATH = os.path.join('sageml', 'meta_learning', 'meta-dataset')


@click.group()
def cli():
    """ Main CLI group """


@cli.command()
@click.option('--dataset_path', required=False, default=PATH, help='Path to the dataset the model will be trained on')
@click.option('--save', required=False, help='Path that the model will be saved on.')
def train(dataset_path: str, save: str):
    """ Trains the model """
    score_dataset = pd.read_csv(os.path.join(dataset_path, 'scores.csv'))
    param_dataset = pd.read_csv(os.path.join(dataset_path, 'parameters.csv'))
    new_model, preprocessor = train_meta_model(score_dataset, param_dataset)
    if save is not None:
        ...


@cli.group()
def dataset():
    """ Main meta-dataset related operations """


@dataset.command()
@click.argument('path', required=True, type=click.Path())
@click.option('--save', type=click.Path(), help='Where to save the file')
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
@click.option('-r', is_flag=True, help='Interprets path as a directory of datasets')
@click.argument('path', required=True, type=click.Path())
def add(r, path):
    """ Adds entry to the dataset """
    # TODO
    if r is True:
        new_dataset = create_datasets(path)
    # else:
    #     new_dataset = _evaluate_score(path)


if __name__ == '__main__':
    from sageml.utils import options
    options.device = 'cpu'
    cli()
