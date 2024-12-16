from typing import Tuple
from prefect import flow
from turbo_ml.workflow import generate_training_parameters, train_meta_model, save_meta_model, evaluate_datasets
from turbo_ml.meta_learning.dataset_parameters import BallMapperFeatures


@flow(name='Full Meta Model Workflow', log_prints=True)
def full_pipeline() -> Tuple[int]:
    training_parameters = generate_training_parameters(
        meta_data_extractor=BallMapperFeatures())
    evaluations = evaluate_datasets()
    model, preprocessor = train_meta_model(training_parameters, evaluations)
    save_meta_model(model, preprocessor, 'new_model')
    return model, preprocessor


if __name__ == '__main__':
    full_pipeline()
