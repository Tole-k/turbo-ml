from typing import Tuple
from prefect import flow
from prefect.task_runners import DaskTaskRunner
from turbo_ml.workflow import generate_training_parameters, load_algorithms_evaluations, train_meta_model, save_meta_model
from turbo_ml.meta_learning.dataset_parameters import SimpleMetaFeatures, BallMapperFeatures
from turbo_ml.workflow.testing import test_TurboML


@flow(name='Full Meta Model Workflow', log_prints=True)
def model() -> Tuple[int]:
    training_parameters = generate_training_parameters.submit(
        meta_data_extractor=SimpleMetaFeatures())
    evaluations = load_algorithms_evaluations.submit()
    
    model, preprocessor = train_meta_model(
        training_parameters, evaluations, epochs=100)
    save_meta_model(model, preprocessor, 'new_model')
    test_TurboML('new_model/', SimpleMetaFeatures())
    return model, preprocessor


if __name__ == '__main__':
    model()
