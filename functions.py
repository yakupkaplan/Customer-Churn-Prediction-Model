# FUNCTIONS FOR CUSTOMER CHURN PREDICTION

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
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, ClassPredictionError
from yellowbrick.model_selection import LearningCurve, FeatureImportances

import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

from collections import Counter
from imblearn.over_sampling import SMOTE

#%config InlineBackend.figure_format = 'retina'

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);


# Function that catches categorical variables, prints the distribution and ratio of unique values and finally creates a countplot
def cats_summary1(data, target):
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in ['Exited']]
    for col in cat_names:
        print(pd.DataFrame({col: data[col].value_counts(),
                            "Ratio": 100 * data[col].value_counts() / len(data),
                            "TARGET_MEAN": data.groupby(col)[target].mean()}), end="\n\n\n")
        sns.countplot(x=col, hue='Exited', data=data)
        plt.show()


# Function to plot histograms for numerical variables
def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function to report variables with outliers and return the names of the variables with outliers with a list
def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


# Function to reassign up/low limits to the ones above/below up/low limits by using apply and lambda method
def replace_with_thresholds_with_lambda(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# Define a function to apply one hot encoding to categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=False, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


# Evaluate each model in turn by looking at train and test errors and scores
def evaluate_classification_model_holdout(models, X_train, X_test, y_train, y_test):

    # Define lists to track names and results for models
    names = []
    train_accuracy_results = []
    test_accuracy_results = []
    test_f1_scores = []
    test_roc_auc_scores = []
    test_precision_scores = []
    test_recall_scores = []
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
        test_f1_score = f1_score(y_test, y_test_pred)
        test_f1_scores.append(test_f1_score)
        test_roc_auc_score = roc_auc_score(y_test, y_test_pred)
        test_roc_auc_scores.append(test_roc_auc_score)
        test_precision_score = precision_score(y_test, y_test_pred)
        test_precision_scores.append(test_precision_score)
        test_recall_score = recall_score(y_test, y_test_pred)
        test_recall_scores.append(test_recall_score)

        names.append(name)
        msg = "%s: Accuracy: %f, ROCAUCScore: %f, F1 Score: %f" % (name, test_accuracy_result, test_roc_auc_score, test_f1_score)
        print(msg)

    print('\n################ Train and test results for the model: ################\n')
    data_result = pd.DataFrame({'models': names,
                                'accuracy_train': train_accuracy_results,
                                'accuracy_test': test_accuracy_results,
                                'roc_auc_score': test_roc_auc_scores,
                                'f1_score_test': test_f1_scores,
                                'precision_test': test_precision_scores,
                                'recall_test': test_recall_scores})
    data_result.set_index('models')
    print(data_result)

    # Plot comparison table for accuracy scores
    plt.figure(figsize=(15, 12))
    sns.barplot(x='accuracy_test', y='models', data=data_result.sort_values(by="accuracy_test", ascending=False), color="r")
    plt.xlabel('Accuracy Scores')
    plt.ylabel('Models')
    plt.title('Accuracy Scores For Test Set')
    plt.show()

    # Plot comparison table for roc_auc_scores
    plt.figure(figsize=(15, 12))
    sns.barplot(x='roc_auc_score', y='models', data=data_result.sort_values(by="roc_auc_score", ascending=False), color="salmon")
    plt.xlabel('ROC AUC Scores')
    plt.ylabel('Models')
    plt.title('ROC AUC Scores For Test Set')
    plt.show()


# Define a function to plot feature_importances
def plot_feature_importances(tuned_model, X_train, X_test, y_train, y_test):
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
def plot_confusion_matrix_yb(model, X_train, X_test, y_train, y_test):
    model_cm = ConfusionMatrix(model, percent=True, classes=['not_churn', 'churn'], cmap='Blues')
    model_cm.fit(X_train, y_train)
    model_cm.score(X_test, y_test)
    model_cm.show();


# Function to plot classification_report by using yellowbrick
def plot_classification_report_yb(model, X_train, X_test, y_train, y_test):
    visualizer = ClassificationReport(model, classes=['not_churn', 'churn'], support=True, cmap='Blues')
    visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show();


# Funtion to plot ROC-AUC Curve
def plot_roc_auc_curve(model, X_train, X_test, y_train, y_test):
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
def plot_roc_auc_curve_yb(model, X_train, X_test, y_train, y_test):
    visualizer = ROCAUC(model, classes=['not_churn', 'churn'])
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show();  # Finalize and show the figure


# Function to plot prediction errors
def plot_class_prediction_error_yb(model, X_train, X_test, y_train, y_test):
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(model, classes=['not_churn', 'churn'])
    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)
    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)
    # Draw visualization
    visualizer.show();


# Function to plot learning curves
def plot_learning_curve(model_tuned, X, y):
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(model_tuned, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4)
    visualizer.fit(X, y)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure


# Function to report results quickly
def report_results_quickly(model, X_train, X_test, y_train, y_test):
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
def plot_results(model, X_train, X_test, y_train, y_test):
    plot_confusion_matrix_yb(model, X_train, X_test, y_train, y_test)
    plot_classification_report_yb(model, X_train, X_test, y_train, y_test)
    plot_roc_auc_curve_yb(model, X_train, X_test, y_train, y_test)
    plot_class_prediction_error_yb(model, X_train, X_test, y_train, y_test)

