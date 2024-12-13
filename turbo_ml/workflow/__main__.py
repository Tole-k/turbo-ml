from prefect import flow
from .extract_parameters import generate_training_parameters
from .train_model import train_meta_model
from turbo_ml.meta_learning.dataset_parameters.topological import BallMapperFeatures

@flow(name='Generate Training Parameters', log_prints=True)
def main():
    training_parameters = generate_training_parameters()
    
    
@flow(name='Train Meta Model', log_prints=True)
def model():
    training_parameters = generate_training_parameters(meta_data_extractor=BallMapperFeatures())
    result = train_meta_model(training_parameters)
    print('Flow created')
    print(result)
    
    
if __name__ == '__main__':
    model()