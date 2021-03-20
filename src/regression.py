

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier


class assess:

    def __init__(self,fast_eval, n_jobs=-1):

        self.n_jobs = n_jobs
        self.fast_eval = fast_eval


class Regressor(assess, estimator= RandomForestClassifier):

    def __init__(self):
        self.estimator = None


    def fit(self, X, Y):

        #scikkit learn pipeline > NA input method(continuous and cat var) standarddize one-hot encode

        preprocessing_steps = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            # ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot encoder', OneHotEncoder(categories='auto',
                                     sparse=False,
                                     handle_unknown='ignore'))])


        clf = RandomForestClassifier(max_depth=2, random_state=0)
        >> > clf.fit(X, y)















