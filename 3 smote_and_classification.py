# CUSTOMER CHURN PREDICTION

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
    - Load the saved dataset
    - Balancing Dataset SMOTE --> Imbalanced data problem will be tried to solved by using SMOTE technique.
    - Modeling
        - Base models: LogisticRegression, GaussianNB, KNeighborsClassifier, SVC, MLPClassifier, DecisionTreeClassifier,
                       BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                       XGBClassifier, LGBMClassifier, CatBoostClassifier, NGBClassifier
    - Model Evaluation
    - Model Tuning
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

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, f1_score, precision_score, recall_score, confusion_matrix
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

from collections import Counter
from imblearn.over_sampling import SMOTE

# Import functions from functions.py.
import functions as fn


# Load the dataset
churn_preprocessed = pd.read_csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\churn_prepared.csv")
df = churn_preprocessed.copy()
df.head()

# Define dependent and independent variables
y = df["Exited"]
X = df.drop(["Exited"], axis=1)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=12345)
print("Shape of X_train: ", X_train.shape, # Shape of X_train:  (8000, 11)
      "\nShape of y_train: ", y_train.shape, # Shape of X_train:  (8000, 11)
      "\nShape of X_test: ", X_test.shape, # Shape of X_test:  (2000, 11)
      "\nShape of y_test: ", y_test.shape) # Shape of y_test:  (2000,)

# summarize class distribution
counter = Counter(y)
print(counter) # Counter({0: 7963, 1: 2037})


# Balance training data set by using SMOTE technique.

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1))) # 1610
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0))) # 6390

# define oversample strategy
oversample = SMOTE(sampling_strategy=0.5) #

# fit and apply the transform
X_over, y_over = oversample.fit_resample(X_train, y_train)

# summarize class distribution
print(Counter(y_over)) # Counter({0: 6370, 1: 3185})

print('After OverSampling, the shape of X_train: {}'.format(X_over.shape)) # (9555, 11)
print('After OverSampling, the shape of y_train: {} \n'.format(y_over.shape)) # (9555,)

print("After OverSampling, counts of label '1': {}".format(sum(y_over==1))) # 3185
print("After OverSampling, counts of label '0': {}".format(sum(y_over==0))) # 6370


# MODELING


# EFFECT OF STANDARDIZATION

# We want to the effect of scaling on the dataset.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# LogisticRegression with make_pipeline
logreg = make_pipeline(StandardScaler(), LogisticRegression())
#fn.evaluate_classification_model_holdout([('LogisticRegression', logreg)], X_train, X_test, y_train, y_test) # 0.81 instead of  0.790667

# KNN with make_pipeline
knn = make_pipeline(RobustScaler(), KNeighborsClassifier())
#fn.evaluate_classification_model_holdout([('KNN', knn)], X_train, X_test, y_train, y_test) # .839333 instead of 0.764667

# SVC with make_pipeline
svc = make_pipeline(RobustScaler(), SVC())
#fn.evaluate_classification_model_holdout([('SVM', svc)], X_train, X_test, y_train, y_test) # 0.855333 instead of 0.792667

# ANN with make_pipeline
ann = make_pipeline(MinMaxScaler(), MLPClassifier())
#fn.evaluate_classification_model_holdout([('ANN', ann)], X_train, X_test, y_train, y_test) # 0.86 instead of 0.790667

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
'''
################ Train and test results for the model: ################
                models  accuracy_train  accuracy_test  roc_auc_score  \
0   LogisticRegression        0.831250         0.8295       0.646934   
1          Naive Bayes        0.790625         0.7930       0.508778   
2                  KNN        0.866250         0.8300       0.662797   
3                  SVM        0.797625         0.7965       0.501829   
4                  ANN        0.875500         0.8610       0.723415   
5                 CART        1.000000         0.7915       0.689848   
6          BaggedTrees        0.984250         0.8515       0.698244   
7                   RF        1.000000         0.8595       0.697779   
8             AdaBoost        0.856500         0.8565       0.707786   
9                  GBM        0.872125         0.8590       0.703867   
10             XGBoost        0.961375         0.8530       0.714735   
11            LightGBM        0.909875         0.8615       0.720985   
12            CatBoost        0.907250         0.8625       0.719784   
13             NGBoost        0.861500         0.8585       0.696236   
    f1_score_test  precision_test  recall_test  
0        0.447326        0.657143     0.339066  
1        0.054795        0.387097     0.029484  
2        0.476923        0.637860     0.380835  
3        0.009732        0.500000     0.004914  
4        0.589971        0.738007     0.491400  
5        0.502980        0.488426     0.518428  
6        0.546565        0.721774     0.439803  
7        0.551834        0.786364     0.425061  
8        0.564492        0.738095     0.457002  
9        0.560748        0.765957     0.442260  
10       0.571429        0.702509     0.481572  
11       0.587183        0.746212     0.484029  
12       0.586466        0.755814     0.479115  
13       0.548644        0.781818     0.422604  

'''


# MODEL TUNING

'''
Models to be tuned:
    - LogisticRegression
    - RandomForestClassifier
    - XGBClassifier
    - LightGBMClassifier
    - CatBoostClassifier     
'''

# LogisticRegression

df["Exited"].value_counts()

logreg_sensitive = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', class_weight={0: 0.2037, 1: 0.7963}))
logreg_sensitive.fit(X_train, y_train)
yhat_sensitive = logreg_sensitive.predict(X_test)

print('F-Measure: %.3f' % f1_score(y_test, yhat_sensitive)) # 0.533
print('Accuracy: %.3f' % accuracy_score(y_test, yhat_sensitive)) # 0.734
print('Precision: %.3f' % precision_score(y_test, yhat_sensitive)) # 0.415
print('Recall: %.3f' % recall_score(y_test, yhat_sensitive)) # 0.747
print('ROC AUC Score: %.3f' % roc_auc_score(y_test, yhat_sensitive)) # 0.739


# RandomForestClassifier

rf_model = RandomForestClassifier(random_state=123456)
rf_params = {"n_estimators": [100, 200, 500, 1000],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 10, 30],
             "max_depth": [3, 5, 8, None]}

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
rf_cv_model.best_params_ # {'max_depth': 8, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 500}

# Final Model
rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred) # 0.859
fn.plot_classification_report_yb(rf_tuned, X_train, X_test, y_train, y_test)

# Visualization of Results --> Feature Importances
fn.plot_feature_importances(rf_tuned, X_train, X_test, y_train, y_test)
fn.report_results_quickly(rf_tuned, X_train, X_test, y_train, y_test)
fn.plot_results(rf_tuned, X_train, X_test, y_train, y_test)
fn.plot_learning_curve(rf_tuned, X, y)


# XGBClassifier

xgb_model = XGBClassifier(random_state=123456)
xgb_params = {"learning_rate": [0.01, 0.1, 0.2, 1],
              "max_depth": [3, 5, 6, 8],
              "subsample": [0.5, 0.9, 1.0],
              "n_estimators": [100, 500, 1000]}

xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgb_cv_model.best_params_ # {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500, 'subsample': 0.5}

# Final Model
xgb_tuned = XGBClassifier(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test, y_pred) # 0.8565
fn.plot_classification_report_yb(xgb_tuned, X_train, X_test, y_train, y_test)

# Visualization of Results --> Feature Importances
fn.plot_feature_importances(xgb_tuned, X_train, X_test, y_train, y_test)
fn.report_results_quickly(xgb_tuned, X_train, X_test, y_train, y_test)
fn.plot_results(xgb_tuned, X_train, X_test, y_train, y_test)
fn.plot_learning_curve(xgb_tuned, X, y)


# LightGBMClassifier

lgbm_model = LGBMClassifier(random_state=123456)
lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8]}

lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
lgbm_cv_model.best_params_ #= {'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 500}

# Final Model
lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred) # 0.8615
fn.plot_classification_report_yb(lgbm_tuned, X_train, X_test, y_train, y_test)

# Visualization of Results --> Feature Importances
fn.plot_feature_importances(lgbm_tuned, X_train, X_test, y_train, y_test)
fn.report_results_quickly(lgbm_tuned, X_train, X_test, y_train, y_test)
fn.plot_results(lgbm_tuned, X_train, X_test, y_train, y_test)
fn.plot_learning_curve(lgbm_tuned, X, y)


# CatBoostClassifier

catb_model = CatBoostClassifier(random_state=123456)
catb_params = {"iterations": ['None', 200, 500],
               "learning_rate": ['None', 0.01, 0.1],
               "depth": ['None', 3, 6]}

catb_cv_model = GridSearchCV(catb_model, catb_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
catb_cv_model.best_params_ # {'depth': 3, 'iterations': 200, 'learning_rate': 0.1}

# Final Model

catb_tuned = CatBoostClassifier(**catb_cv_model.best_params_).fit(X_train, y_train)
y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test, y_pred) # 0.8575
fn.plot_roc_auc_curve(catb_tuned, X_train, X_test, y_train, y_test)

# Visualization of Results --> Feature Importances
fn.plot_feature_importances(catb_tuned, X_train, X_test, y_train, y_test)
fn.report_results_quickly(catb_tuned, X_train, X_test, y_train, y_test)
fn.plot_results(catb_tuned)
fn.plot_learning_curve(catb_tuned, X, y)


# Comparison of tuned models

tuned_models = [('RF', rf_tuned),
                ('XGBoost', xgb_tuned),
                ('LightGBM', lgbm_tuned),
                ('CatBoost', catb_tuned)]


fn.evaluate_classification_model_holdout(tuned_models, X_train, X_test, y_train, y_test)


# Cross Validation Scores for the best model: LightGBMClassifier

# validasyon error, accuracy score, confusion matrix
cv_results_accuracy = cross_val_score(lgbm_tuned, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_accuracy.mean()) # 0.8612499999999998

# import sklearn.metrics
# sklearn.metrics.SCORERS.keys()
# dict_keys(['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error', 'neg_mean_absolute_error',
# 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_root_mean_squared_error', 'neg_mean_poisson_deviance',
# 'neg_mean_gamma_deviance', 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted',
# 'balanced_accuracy', 'average_precision', 'neg_log_loss', 'neg_brier_score', 'adjusted_rand_score', 'homogeneity_score',
# 'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score',
# 'fowlkes_mallows_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted',
# 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
# 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted'])

cv_results_roc_auc_score = cross_val_score(lgbm_tuned, X_train, y_train, cv=10, scoring="roc_auc")
print("cross_val_score(train):", cv_results_roc_auc_score.mean()) # 0.8563743005460797

cv_results_accuracy = cross_val_score(lgbm_tuned, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_accuracy.mean()) # 0.86

cv_results_roc_auc_score = cross_val_score(lgbm_tuned, X_test, y_test, cv=10, scoring="roc_auc")
print("cross_val_score(test):", cv_results_roc_auc_score.mean()) # 0.8352496692360791

y_train_pred = lgbm_tuned.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred)) # 0.8835
print("accuracy_score(test):", accuracy_score(y_test, y_pred)) # 0.8575
print("roc_auc_score(train):", roc_auc_score(y_train, y_train_pred)) # 0.7462944592655373
print("roc_auc_score(test):", roc_auc_score(y_test, y_pred)) # 0.7065840879400201
print(classification_report(y_test, y_pred))

# Confusion matrix for the final model
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()
