from turbo_ml.forecast.dataset_characteristics import StatisticalParametersExtractor, DatasetDescription
from datasets import get_iris, get_breast_cancer


def test_StatisticalParametersExtractor():
    dataset, target = get_breast_cancer()

    descriptor = StatisticalParametersExtractor(dataset, target)
    description = descriptor.describe_dataset()
    assert isinstance(description, DatasetDescription)
    assert description.num_rows > 0
    assert description.num_columns > 0
    assert description.target_nans >= 0
    assert description.task in ["classification", "regression"]
    assert description.highest_correlation >= description.lowest_correlation
    assert description.highest_eigenvalue >= description.lowest_eigenvalue
    assert description.number_of_highly_correlated_features >= 0
    assert description.number_of_lowly_correlated_features >= 0


def test_data_description():
    dataset, target = get_iris()

    descriptor = StatisticalParametersExtractor(dataset, target)
    description = descriptor.describe_dataset()
    assert description.num_rows == 150
    assert description.num_columns == 4
    assert description.target_nans == 0
    assert description.task == "classification"
    assert description.num_classes == 3
    assert description.highest_correlation >= description.lowest_correlation
    assert description.highest_eigenvalue >= description.lowest_eigenvalue
    assert description.smallest_class_freq == description.biggest_class_freq
