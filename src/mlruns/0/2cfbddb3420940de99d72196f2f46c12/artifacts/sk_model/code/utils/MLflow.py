import mlflow
import mlflow.sklearn
import logging
import mlflow.pyfunc
import os
def track(
    metrics=None,
    params=None,
    artifacts=None,
    model=None,
    feature_engineering=None,
    preprocessing=None,
    mlflow_dir= './mlruns',
    artifacts_path = "outputs/plots/mlflow_artifacts"):
    
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(mlflow_dir)
    with mlflow.start_run():

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mlflow.log_artifacts(artifacts_path)
                
        ml_app = MLApplication(model, preprocessing)
        # print(os.getcwd())
        mlflow.pyfunc.log_model(python_model = ml_app, # Specify the model to log
                                artifact_path = "sk_model", # where to log it
                                code_path = ['utils'], # local dependances
                               conda_env = '../config/conda.yml') 
#         mlflow.sklearn.log_model(model, "sk_models")   


class MLApplication(mlflow.pyfunc.PythonModel):

    def __init__(self, model, preprocessing):
        self.model = model
#         self.feature_engineering = feature_engineering
        self.preprocessing = preprocessing

    def predict(self, context, x):
        
#         x = self.feature_engineering(x)
        x = self.preprocessing.transform(x, verbose=False)
        return self.model.predict(x)