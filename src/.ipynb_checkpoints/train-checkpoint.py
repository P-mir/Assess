import argparse
from utils.MLutils import Preprocess, Models, explain
from utils.MLflow import track
import pandas as pd
pd.set_option('display.max_rows', 500)
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

def run(df,explainmodel=False):

    X = df.drop('label', axis=1)
    Y = df.label

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

    my_model = Model.getModel(task = "classification",n_jobs=1)
    model = my_model.fit(x_train_preprocessed,y_train)
    y_pred = model.predict(x_test_preprocessed)

    metrics, params, artifacts = my_model.compute_metrics_and_graphs(y_pred, y_test,
                                                                     output_path="outputs/plots/mlflow_artifacts")
    if explainmodel==True:
        print("explain model' predictions...")
        explain(x_train_preprocessed, model, path="outputs/plots/shap")

    track(metrics, params, artifacts, model, preprocessing,
          mlflow_dir='./mlruns',
        artifacts_path="outputs/plots/mlflow_artifacts"
          )


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(prog='train', description='Train the model')

    # Add the arguments
    my_parser.add_argument('--path', type=str, help='path to data')
    my_parser.add_argument('--explainmodel',dest='explain', type=bool, nargs='?', const=True, default=False,
                           help='whether to explain model prediction (boolean)')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    path = args.path
    explainmodel = args.explain
    # df = pd.read_csv('../data/iris2.csv', index_col=False, header=0)
    df = pd.read_csv(path, index_col=False, header=0)
    run(df, explainmodel=explainmodel)
    print("model trained")
