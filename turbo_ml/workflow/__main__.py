from prefect import flow
from turbo_ml.workflow import generate_training_parameters, load_algorithms_evaluations, train_meta_model
from turbo_ml.meta_learning.dataset_parameters import SimpleMetaFeatures, BallMapperFeatures
    
@flow(name='Full Meta Model Workflow', log_prints=True)
def model():
    training_parameters = generate_training_parameters(meta_data_extractor=BallMapperFeatures())
    evaluations = load_algorithms_evaluations()
    result = train_meta_model(training_parameters, evaluations)
    print('Flow created')
    print(result)
    
    
if __name__ == '__main__':
    model()