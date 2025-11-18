import mlflow

def setup_mlflow(experiment="fraud_detection"):
    mlflow.set_tracking_uri("file:./mlruns") #track log (params...) of models trained
    mlflow.set_experiment(experiment) #start experiment fraud_detection
    return mlflow