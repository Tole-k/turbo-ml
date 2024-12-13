from prefect import flow
from turbo_ml.workflow import generate_training_parameters, load_algorithms_evaluations, train_meta_model, save_meta_model
from turbo_ml.meta_learning.dataset_parameters import SimpleMetaFeatures, BallMapperFeatures

@flow(name='Full Meta Model Workflow', log_prints=True)
def model():
    training_parameters = generate_training_parameters(meta_data_extractor=SimpleMetaFeatures())
    evaluations = load_algorithms_evaluations()
    model, preprocessor = train_meta_model(training_parameters, evaluations, epochs=100)
    save_meta_model(model, preprocessor, 'new_model')
    return model, preprocessor
    
    
if __name__ == '__main__':
    model()