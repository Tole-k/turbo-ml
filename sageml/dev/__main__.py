""" CLI tool for developers """
import click

from sageml.dev.functions import (
    _read_dataset,
    create_dataset,
)


@click.group()
def cli():
    """ Main CLI group """


@cli.command()
@click.option('--dataset_path', required=False, help='Path to the dataset the model will be trained on')
@click.option('--save', required=False, help='Path that the model will be saved on.')
def train(dataset_path: str, save: str):
    """ Trains the model """
    # TODO


@cli.group()
def dataset():
    """ Main meta-dataset related operations """


@dataset.command()
@click.argument('path', required=True, type=click.Path())
@click.option('--save', type=click.Path(), help='Where to save the file')
def create(path: str, save: str | None):
    """ Creates meta-dataset """
    click.echo('Creating new dataset!')
    new_dataset = create_dataset(path)
    click.echo('Dataset created successfully!')
    print(new_dataset)
    if save is not None:
        click.echo('Saving dataset ...')
        new_dataset.to_csv(save)


@dataset.command()
@click.option('-r', is_flag=True, help='Interprets path as a directory of datasets')
@click.argument('path', required=True, type=click.Path())
def add(r, path):
    """ Adds entry to the dataset """
    # TODO
    if r is True:
        new_dataset = create_dataset(path)
    else:
        new_dataset = _read_dataset(path)


if __name__ == '__main__':
    cli()
