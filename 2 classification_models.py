# CUSTOMER CHURN PREDICTION

# CLASSIFICATION MODELS - HOLDOUT

'''
Problem:
    Can you develop a machine learning model that can predict customers who will leave the company (customer churn)?
    The aim is to predict whether a bank's customers leave the bank or not.
    The situation that defines the customer churn is the customer closing his bank account.

Data Set Story:
    It consists of 10000 observations and 12 variables.
    Independent variables contain information about customers.
    The dependent variable expresses the customer churn situation.

Variables:
    Surname,CreditScore, Geography (Country),Gender (Female / Male), Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    Exited: Churn or not? (0 = No, 1 = Yes)

Steps to follow:
    - SMOTE --> Imbalanced data
    - Load the saved dataset
    - General View
    - Modeling
        - Base models: LogisticRegression, GaussianNB, KNeighborsClassifier, SVC, MLPClassifier, DecisionTreeClassifier,
                       BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                       XGBClassifier, LGBMClassifier, CatBoostClassifier, NGBClassifier
    - Model Evaluation
    - Effect of Scaling on the Model Performance
    - Model Results
'''


# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier

import os
import pickle

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, ClassPredictionError
from yellowbrick.model_selection import LearningCurve, FeatureImportances

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

#%config InlineBackend.figure_format = 'retina'

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);

# Import functions from functions.py.
import functions as fn


# Load the dataset
churn_preprocessed = pd.read_csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\churn_prepared.csv")
df = churn_preprocessed.copy()


## GENERAL VIEW

df.head()
df.shape
df.info()
df.columns
df.index
df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# Check for missing values
df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False)


# MODELING

# Define dependent and independent variables
y = df["Exited"]
X = df.drop(["Exited"], axis=1)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12345)

# See the results for base models
base_models = [('LogisticRegression', LogisticRegression()),
               ('Naive Bayes', GaussianNB()),
               ('KNN', KNeighborsClassifier()),
               ('SVM', SVC()),
               ('ANN', MLPClassifier()),
               ('CART', DecisionTreeClassifier()),
               ('BaggedTrees', BaggingClassifier()),
               ('RF', RandomForestClassifier()),
               ('AdaBoost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ("XGBoost", XGBClassifier()),
               ("LightGBM", LGBMClassifier()),
               ("CatBoost", CatBoostClassifier(verbose=False)),
               ("NGBoost", NGBClassifier(verbose=False))]

fn.evaluate_classification_model_holdout(base_models, X_train, X_test, y_train, y_test)

# Take a look at the confusion matrix for clearer view of the results.
for name, model in base_models:
    print(model)
    fn.plot_classification_report_yb(model, X_train, X_test, y_train, y_test)


# Effect of Scaling on the Model Performance

# We want to see the effect of scaling on the dataset.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# LogisticRegression with make_pipeline
logreg = make_pipeline(StandardScaler(), LogisticRegression())
fn.evaluate_classification_model_holdout([('LogisticRegression', logreg)], X_train, X_test, y_train, y_test) # 0.81 instead of  0.790667

# KNN with make_pipeline
knn = make_pipeline(RobustScaler(), KNeighborsClassifier())
fn.evaluate_classification_model_holdout([('KNN', knn)], X_train, X_test, y_train, y_test) # .839333 instead of 0.764667

# SVC with make_pipeline
svc = make_pipeline(RobustScaler(), SVC())
fn.evaluate_classification_model_holdout([('SVM', svc)], X_train, X_test, y_train, y_test) # 0.855333 instead of 0.792667

# ANN with make_pipeline
ann = make_pipeline(MinMaxScaler(), MLPClassifier())
fn.evaluate_classification_model_holdout([('ANN', ann)], X_train, X_test, y_train, y_test) # 0.86 instead of 0.790667

# See the results for base models after scaling, again.

base_models = [('LogisticRegression', logreg),
               ('Naive Bayes', GaussianNB()),
               ('KNN', knn),
               ('SVM', svc),
               ('ANN', ann),
               ('CART', DecisionTreeClassifier()),
               ('BaggedTrees', BaggingClassifier()),
               ('RF', RandomForestClassifier()),
               ('AdaBoost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ("XGBoost", XGBClassifier()),
               ("LightGBM", LGBMClassifier()),
               ("CatBoost", CatBoostClassifier(verbose=False)),
               ("NGBoost", NGBClassifier(verbose=False))]

fn.evaluate_classification_model_holdout(base_models, X_train, X_test, y_train, y_test)

# ################ Train and test results for the model: ################
#                 models  accuracy_train  accuracy_test  precision_test  \
# 0   LogisticRegression        0.814250         0.7990        0.588652
# 1          Naive Bayes        0.787875         0.7740        0.345679
# 2                  KNN        0.878375         0.8280        0.664032
# 3                  SVM        0.865000         0.8465        0.826087
# 4                  ANN        0.868875         0.8570        0.780876
# 5                 CART        1.000000         0.7875        0.502358
# 6          BaggedTrees        0.984375         0.8440        0.720307
# 7                   RF        1.000000         0.8520        0.767347
# 8             AdaBoost        0.860000         0.8500        0.755020
# 9                  GBM        0.876000         0.8560        0.779116
# 10             XGBoost        0.957750         0.8465        0.715827
# 11            LightGBM        0.916875         0.8515        0.751938
# 12            CatBoost        0.910125         0.8610        0.787645
# 13             NGBoost        0.864250         0.8550        0.824645
#     recall_test  f1_score_test
# 0      0.194379       0.292254
# 1      0.065574       0.110236
# 2      0.393443       0.494118
# 3      0.355972       0.497545
# 4      0.459016       0.578171
# 5      0.498829       0.500588
# 6      0.440281       0.546512
# 7      0.440281       0.559524
# 8      0.440281       0.556213
# 9      0.454333       0.573964
# 10     0.466042       0.564539
# 11     0.454333       0.566423
# 12     0.477752       0.594752
# 13     0.407494       0.545455


