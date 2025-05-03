""" Dataset generation """
from sageml.workflow import (generate_training_parameters, train_meta_model,
                             load_algorithms_evaluations)
from sageml.meta_learning.dataset_parameters.sota import sota_meta_features
from sageml.utils import options


def full_pipeline() -> tuple:
    evaluations = load_algorithms_evaluations('algorithm_results.csv')
    training_parameters = generate_training_parameters(output_path='', meta_data_extractor=sota_meta_features(options.meta_features))
    model, preprocessor = train_meta_model(training_parameters, evaluations, 3000)
    return model, preprocessor


if __name__ == '__main__':
    full_pipeline()
