import pytest
import pandas as pd
import numpy as np
import train
import unittest
from utils.MLutils import Preprocess
from parameterized import parameterized

df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
print(df.head(10))

df["Cat1"] = np.random.choice(['First','Second','Third'], 100, replace=True)
df["Cat2"] = np.random.choice([100,200,300, np.nan], 100, replace=True)
df["label"] = np.random.choice(["Target1","Target2","Target3"], 100, replace=True)
X = df.drop('label', axis=1)
Y = df.label

# @pytest.mark.parametrize("X","Y", [X,Y])

class TestsMethods(unittest.TestCase):

    def test_train_test_split(self):
        from sklearn.model_selection import train_test_split
        print(df.head(100))
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.assertTrue((len(x_train)+len(x_test))==len(X))
        self.assertTrue((len(y_train)+len(y_test))==len(Y))
        self.assertTrue(len(x_train) == len(y_train))

    def test_preprocessing(self):

        print(len(df.isna().sum()))
        preprocessing = Preprocess(
                                     numeric_na_fill_method= 'median',
                                     category_na_fill_method='mode')
        df_preprocessed = preprocessing.fit_transform(X)
        self.assertTrue(df_preprocessed.isna().sum().sum()==0)


if __name__ == '__main__':
    unittest.main()


