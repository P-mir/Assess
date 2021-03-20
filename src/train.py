import argparse
from utils.MLutils import Preprocess, Model, explain
from utils.MLflow import track

import pandas as pd
pd.set_option('display.max_rows', 500)
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

def fit(df,features,target,task ='classificaiton', explainmodel=False):
    """  
    1 - Preprocess the data: impute missing value, normalize if necessary, one-hot encode
    2 - Instanciate  and run a training pipeline performing a grid search along with cross-validation  

    Args:
        df (DataFrame): 
        features (list(str)): Fea  tures to train the model on
        target (str): Target va  variable
        task (str, optional): Defaults to 'classificaiton'.
        explainmodel (bool, optional): Whether to explain model

    Returns:
    """


    X = df[features]
    Y = df[target]

    # X = feature_engineering(X,
    #                     add_missing_flags = True # useful if missing value are not MCAR
    #                     )

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #create the pipeline and make the transform
    preprocessing = Preprocess(
    #     scaler = 'normalize',
        numeric_na_fill_method = 'median',
        category_na_fill_method = 'mode'
        )
    print("x_train shape: ", x_train.shape)
    x_train_preprocessed = preprocessing.fit_transform(x_train)
    x_test_preprocessed = preprocessing.transform(x_test, verbose=False)

    my_model = Model.getModel(task = task,n_jobs=1)
    model = my_model.fit(x_train_preprocessed,y_train)
    y_pred = model.predict(x_test_preprocessed)

    metrics, params, artifacts = my_model.compute_metrics_and_graphs(y_pred, y_test,
                                                                     output_path="outputs/plots/mlflow_artifacts")
    if explainmodel:
        print("explain model' predictions...")
        explain(x_train_preprocessed, model,task=task)

    track(metrics, params, artifacts, model, preprocessing,
          mlflow_dir='./mlruns',
        artifacts_path="outputs/plots/mlflow_artifacts"
          )
    return model, preprocessing, X

def predict(model, preprocessing, x):
    """
    Args:
        model ([type]): Trained model
        preprocessing ([type]): Fiited preprocessing pipeline
        x (DataFrame): Input data

    Returns:
        [list]: [description]
    """

    x = preprocessing.transform(x, verbose=False)
    return model.predict(x)


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(prog='train', description='Train the model')

    # Add the arguments
    my_parser.add_argument('--target', type=str, help='Response variable')
    my_parser.add_argument('--path', type=str, help='Path to data')
    my_parser.add_argument('--explainmodel',dest='explain', type=bool, nargs='?', const=True, default=False,
                           help='whether to explain model prediction (boolean)')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    target = args.target
    path = args.path
    explainmodel = args.explain
    # df = pd.read_csv('../data/iris2.csv', index_col=False, header=0)
    df = pd.read_csv(path, index_col=False, header=0)
    fit(df,target, explainmodel=explainmodel)
    print("model trained")
