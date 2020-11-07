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

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import preprocessing

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, ClassPredictionError
from yellowbrick.model_selection import LearningCurve, FeatureImportances

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%config InlineBackend.figure_format = 'retina'

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);


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


# Evaluate each model in turn by looking at train and test errors and scores
def evaluate_classification_model_holdout(models):

    # Define lists to track names and results for models
    names = []
    train_accuracy_results = []
    test_accuracy_results = []
    test_precision_scores = []
    test_recall_scores = []
    test_f1_scores = []
    supports = []

    print('################ Accuracy scores for test set for the models: ################\n')
    for name, model in models:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy_result = accuracy_score(y_train, y_train_pred)
        test_accuracy_result = accuracy_score(y_test, y_test_pred)
        train_accuracy_results.append(train_accuracy_result)
        test_accuracy_results.append(test_accuracy_result)
        test_precision_score = precision_score(y_test, y_test_pred)
        test_precision_scores.append(test_precision_score)
        test_recall_score = recall_score(y_test, y_test_pred)
        test_recall_scores.append(test_recall_score)
        test_f1_score = f1_score(y_test, y_test_pred)
        test_f1_scores.append(test_f1_score)

        names.append(name)
        msg = "%s: %f" % (name, test_accuracy_result)
        print(msg)

    print('\n################ Train and test results for the model: ################\n')
    data_result = pd.DataFrame({'models': names,
                                'accuracy_train': train_accuracy_results,
                                'accuracy_test': test_accuracy_results,
                                'f1_score_test': test_f1_scores,
                                'precision_test': test_precision_scores,
                                'recall_test': test_recall_scores})
    data_result.set_index('models')
    print(data_result)

    # Plot the results
    plt.figure(figsize=(15, 12))
    sns.barplot(x='accuracy_test', y='models', data=data_result.sort_values(by="accuracy_test", ascending=False), color="r")
    plt.xlabel('Accuracy Scores')
    plt.ylabel('Models')
    plt.title('Accuracy Scores For Test Set')
    plt.show()


# Define a function to plot feature_importances
def plot_feature_importances(tuned_model):
    feature_importances = pd.DataFrame({'Importance': tuned_model.feature_importances_ * 100, 'Feature': X_train.columns})
    plt.figure()
    sns.barplot(x="Importance", y="Feature", data=feature_importances.sort_values(by="Importance", ascending=False))
    plt.title('Feature Importance - ')
    plt.show()


# Function to plot confusion_matrix
def plot_confusion_matrix(model, X_test, y_test, normalize=True):
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    plt.show()


# Function to plot confusion_matrix
def plot_confusion_matrix_yb(model):
    model_cm = ConfusionMatrix(model, percent=True, classes=['not_churn', 'churn'], cmap='Blues')
    model_cm.fit(X_train, y_train)
    model_cm.score(X_test, y_test)
    model_cm.show();


# Function to plot classification_report by using yellowbrick
def plot_classification_report_yb(model):
    visualizer = ClassificationReport(model, classes=['not_churn', 'churn'], support=True, cmap='Blues')
    visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show();


# Funtion to plot ROC-AUC Curve
def plot_roc_auc_curve(model):
    model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# Funtion to plot ROC-AUC Curve by using yellowbrick
def plot_roc_auc_curve_yb(model):
    visualizer = ROCAUC(model, classes=['not_churn', 'churn'])
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show();  # Finalize and show the figure


# Function to plot prediction errors
def plot_class_prediction_error_yb(model):
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(model, classes=['not_churn', 'churn'])
    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)
    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)
    # Draw visualization
    visualizer.show();


# Function to plot learning curves
def plot_learning_curve(model_tuned):
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(model_tuned, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4)
    visualizer.fit(X, y)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure


# Function to report results quickly
def report_results_quickly(model):
    fig, axes = plt.subplots(2, 2)
    model = model
    visualgrid = [FeatureImportances(model, ax=axes[0][0]),
                  ConfusionMatrix(model, ax=axes[0][1]),
                  ClassificationReport(model, ax=axes[1][0]),
                  ROCAUC(model, ax=axes[1][1])]
    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    plt.show()


# Function to plot all the results
def plot_results(model):
    plot_confusion_matrix_yb(model)
    plot_classification_report_yb(model)
    plot_roc_auc_curve_yb(model)
    plot_class_prediction_error_yb(model)


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

evaluate_classification_model_holdout(base_models)

# Take a look at the confusion matrix for clearer view of the results.
for name, model in base_models:
    print(model)
    plot_classification_report_yb(model)


# We want to the effect of scaling on the dataset.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# LogisticRegression with make_pipeline
logreg = make_pipeline(StandardScaler(), LogisticRegression())
evaluate_classification_model_holdout([('LogisticRegression', logreg)]) # 0.81 instead of  0.790667

# KNN with make_pipeline
knn = make_pipeline(RobustScaler(), KNeighborsClassifier())
evaluate_classification_model_holdout([('KNN', knn)]) # .839333 instead of 0.764667

# SVC with make_pipeline
svc = make_pipeline(RobustScaler(), SVC())
evaluate_classification_model_holdout([('SVM', svc)]) # 0.855333 instead of 0.792667

# ANN with make_pipeline
ann = make_pipeline(MinMaxScaler(), MLPClassifier())
evaluate_classification_model_holdout([('ANN', ann)]) # 0.86 instead of 0.790667

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

evaluate_classification_model_holdout(base_models)

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


