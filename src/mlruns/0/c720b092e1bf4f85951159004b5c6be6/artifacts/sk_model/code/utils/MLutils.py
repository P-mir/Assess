
import pandas as pd
import numpy as np
import os
import shutil

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib. pyplot as plt

import shap


class Preprocess:
    
    def __init__(
        self,
        scaler = None,
        numeric_na_fill_method = None,
        category_na_fill_method = None,
    ):
        self.scaler = scaler
        self.num_na_fill_method = numeric_na_fill_method
        self.cat_na_fill_method = category_na_fill_method

        
    
    def fit(self, df):
        
        self.mean_scaler = dict()
        self.std_scaler = dict()
        self.mode = dict()
        self.median = dict()
        self.numeric_columns = df.select_dtypes(['number']).columns.tolist()
        self.categorical_columns = df.select_dtypes(['object']).columns.tolist()
        self.constant_columns = df.columns[~(df != df.iloc[0]).any()].tolist()
        
        df = df.copy(deep=True)
        
        
        for col in self.numeric_columns:
            if self.scaler == 'normalize':
                self.mean_scaler[col] = df[col].mean() # save parameters
                self.std_scaler[col] = df[col].std() 
            
            if  self.num_na_fill_method=='median':
                self.median[col] = df[col].median()
        
        for col in self.categorical_columns:
            if self.cat_na_fill_method == 'mode':
                self.mode[col] = df[col].mode() # save parameters
                
        return
                
    def transform(self, df, verbose=True):
        
        df = df.copy(deep=True)
        self.verbose = verbose
            
        ##### Remove constant variables #####
        if len(self.constant_columns)>0 and self.verbose==True:
            print("Removed zero variance variables: {}".format(self.constant_columns))
        df = df.drop(self.constant_columns, axis=1)
        
        ##
        self.numeric_columns = df.select_dtypes(['number']).columns.tolist()
        self.categorical_columns = df.select_dtypes(['object']).columns.tolist()  
        
        if self.verbose == True:
            print("Numeric columns:", self.numeric_columns)
            print("categorical columns:", self.categorical_columns)
        ##    
            
        ##### Impute missing values ######
        for col in self.numeric_columns:

            if  self.num_na_fill_method=='median':  # Median is prefered to mean for skewed distribution.      
                n_missing = df[col].isna().sum()  
                if n_missing>0 and self.verbose==True:
                    print("column {}: {} values imputed.".format(col,n_missing))    
                df[col] = df[col].fillna(self.median[col])

        for col in self.categorical_columns:

            if self.cat_na_fill_method=='mode':
                n_missing = df[col].isna().sum()
                if n_missing>0 and self.verbose==True:
                    print("column {}: {} values imputed.".format(col,n_missing))
                df[col] = df[col].fillna(self.mode[col])
        
        ###### Scale ######
                    
        if self.scaler == 'normalize':
            if self.verbose==True:
                print("{} variables normalized.".format(len(self.numeric_columns)))
            for col in self.numeric_columns:
                df[col] = (df[col]-self.mean_scaler[col])/self.std_scaler[col]
        
    
        #####  One hot encode #####
        df = pd.get_dummies(df)
        
        return df 
    
    def fit_transform(self, df):
        
        self.fit(df)
        return self.transform(df)


def feature_engineering(df, add_missing_flags = True):
    """
    Standard feature engineering
    """
    data = df.copy(deep=True)
    
    ### Missing value flags ; if a missing value is not Missing Completely at Random (MCAR), then it makes sense to use this information.
    if add_missing_flags:
        for col in data.columns:
            data[col+"_missing"] = data[col].isnull().map({False:0, True: 1})
    
    return data 


class Models:
    
    def __init__(self,n_jobs=-1):
        self.n_jobs = n_jobs
        
        
    def fit(self,x,y):
        
        self.x = x
        # check type of y to choose the class of problem and model to apply
        
        n_trees = max(200,len(x.columns))
        
#         if len(x)<1000:
#             grid = [{'max_depth': [1,2,3,4,5,10, None], 'n_estimators': [n_trees]}]
#         else:
#             grid = [{'max_depth': [5,10, None], 'n_estimators': [n_trees]}]
        
        grid = [{'max_depth': [1], 'n_estimators': [50]}]
        
        self.model = GridSearchCV(RandomForestClassifier(n_jobs=self.n_jobs), grid)
        self.fitted_cv_model = self.model.fit(x,y)
        
        self.hyperparameters = self.fitted_cv_model.best_params_
        self.validation_accuracy = self.fitted_cv_model.best_score_
        self.fitted_model = self.fitted_cv_model.best_estimator_
        return self.fitted_model
    
    
    def compute_metrics_and_graphs(self, pred, actual,output_path = "outputs/plots/mlflow_artifacts"):
        
        metrics = dict()
        metrics['precision_test'] = precision_score(actual, pred, average='micro')
        metrics['recall_test'] = recall_score(actual, pred, average='micro')
        metrics['accuracy_test'] = accuracy_score(actual, pred)
        metrics['accuracy_validation'] = self.validation_accuracy
#         metrics['auc (test)'] = roc_auc_score(actual, pred)
        
        params = dict()
        params.update(self.hyperparameters)
        
        artifacts = dict()
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save variables importances
        features = self.x.columns.tolist()
        importances = self.fitted_model.feature_importances_
        indices = np.argsort(importances)
        plt.ioff()
        fig = plt.figure()
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        fig.savefig('{}/variable_importances'.format(output_path), bbox_inches="tight")
        
        artifacts['variable_importances'] = 'variable_importances.png'
        
        return metrics, params, artifacts
        
        
#         fpr, tpr, tresholds = roc_curve(y_test[:, i], y_score[:, i])
#         import seaborn as sns; sns.set()
#         import matplotlib.pyplot as plt
#         ax = sns.lineplot(x=fpr, y=tpr,hue=['False positive rate','True positive rate'])
#         ax.savefig("plots/artifacts/roc_curve.png")
    

        
        
#         training_time
        
        
        # To do: add metrics specific to multiclass problems
        
        #balanced accuracy (average of recall obtained on each class)accuracy_score, precision_score, recall_score, roc_auc_score
    

        
        
    
#     def evaluate(self, y_pred, y_test):
#         # Note that for small data the data split between  trai,n validation and test set will underestimate the performance,
#         #in this case simple crossval or even train test split should be prefered.
#         return accuracy_score(y_pred, y_test)
    
    
    
# import lightgbm as lgb
# import neptune
# import neptunecontrib.monitoring.optuna as opt_utils
# import optuna
# import pandas as pd
# from sklearn.model_selection import train_test_split

# N_ROWS = 10000
# TRAIN_PATH = '../data/train.csv'
# NUM_BOOST_ROUND = 300
# EARLY_STOPPING_ROUNDS = 30
# STATIC_PARAMS = {'boosting': 'gbdt',
#                  'objective': 'binary',
#                  'metric': 'auc',
#                  }
# N_TRIALS = 100

# data = pd.read_csv(TRAIN_PATH, nrows=N_ROWS)
# X = data.drop(['ID_code', 'target'], axis=1)
# y = data['target']


# def train_evaluate(X, y, params):
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1234)
#     train_data = lgb.Dataset(X_train, label=y_train)
#     valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
#     model = lgb.train(params, train_data,
#                       num_boost_round=NUM_BOOST_ROUND,
#                       early_stopping_rounds=EARLY_STOPPING_ROUNDS,
#                       valid_sets=[valid_data],
#                       valid_names=['valid'])
#     score = model.best_score['valid']['auc']
#     return score


# def objective(trial):
#     params = {'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
#               'max_depth': trial.suggest_int('max_depth', 1, 30),
#               'num_leaves': trial.suggest_int('num_leaves', 2, 100),
#               'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
#               'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
#               'subsample': trial.suggest_uniform('subsample', 0.1, 1.0)}
#     all_params = {**params, **STATIC_PARAMS}
#     return train_evaluate(X, y, all_params)


# neptune.init('jakub-czakon/blog-hpo')
# neptune.create_experiment(name='optuna sweep')
# monitor = opt_utils.NeptuneMonitor()

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=N_TRIALS, callbacks=[monitor])
# opt_utils.log_study(study)

# neptune.stop()
# view rawhpo40.py hosted with ❤ by GitHub

def explain(x, model, path = "outputs/plots/shap"):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x, check_additivity=False)
    features_names = x.columns

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs("{}/summary_plots".format(path))
    os.makedirs("{}/dependance_plots".format(path))
    os.makedirs("{}/interaction_plots".format(path))

    plt.rcParams.update({'figure.max_open_warning': 0})

    ## Summary plots
    shap.summary_plot(shap_values, x, class_names=model.classes_, show=False)
    plt.savefig("{}/summary_plots/main_summary_plot.png".format(path),dpi=150, bbox_inches='tight')
    plt.clf()
    
    for i, klass in enumerate(model.classes_):
        shap.summary_plot(shap_values[i], x,class_names=model.classes_, show=False)
        plt.savefig("{}/summary_plots/{}_summary_plot.png".format(path, klass),dpi=150, bbox_inches='tight')
        plt.clf()
        
    ## Dependance plots
    for feature in features_names:
        
        for i, klass in enumerate(model.classes_):
            shap.dependence_plot(feature, shap_values[i], x,
                             title='Impact of the {} variable on the prediction of {}'.format(feature,klass),
                             show=False)
            plt.savefig("{}/dependance_plots/{}_{}_dependance_plot.png".format(path, klass,feature),dpi=150, bbox_inches='tight')
            plt.clf()
    ## Interaction plots
    #  TO DO: takes the 5 most imporant values and watch for interaction
#     plt.clf()
#     explainer.shap_interaction_values(x)
#     plt.savefig("plots/shap/interaction_plots/interaction_plot.png",dpi=150, bbox_inches='tight')