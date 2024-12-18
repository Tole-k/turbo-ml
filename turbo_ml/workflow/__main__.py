from typing import Tuple
from prefect import flow
from turbo_ml.workflow import (generate_training_parameters, train_meta_model,save_meta_model,
                               evaluate_datasets, load_algorithms_evaluations, test_TurboML)
from turbo_ml.meta_learning.dataset_parameters import BallMapperFeatures, SimpleMetaFeatures
from sys import argv

@flow(name='Full Meta Model Workflow', log_prints=True)
def full_pipeline() -> Tuple[int]:
    evaluations = load_algorithms_evaluations('algorithm_results.csv')
    training_parameters = generate_training_parameters(
        meta_data_extractor=BallMapperFeatures())
    model, preprocessor = train_meta_model(training_parameters, evaluations, 3000)
    save_meta_model(model, preprocessor, 'new_model')
    test_TurboML('new_model/', SimpleMetaFeatures())
    return model, preprocessor


if __name__ == '__main__':
    full_pipeline()
