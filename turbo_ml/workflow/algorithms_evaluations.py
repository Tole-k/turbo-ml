import pandas as pd
import os
from prefect import task


@task(name='Load Algorithm Evaluations')
def load_algorithms_evaluations():
    return pd.read_csv(os.path.join('datasets', 'results_algorithms.csv'))
