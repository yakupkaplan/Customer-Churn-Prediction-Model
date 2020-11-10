# CUSTOMER CHURN PREDICTION

'''
SMOTE:

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

from collections import Counter
from imblearn.over_sampling import SMOTE


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

# Balance training data set

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


# STANDARDIZATION

# We want to the effect of scaling on the dataset.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# LogisticRegression with make_pipeline
logreg = make_pipeline(StandardScaler(), LogisticRegression())
#evaluate_classification_model_holdout([('LogisticRegression', logreg)]) # 0.81 instead of  0.790667

# KNN with make_pipeline
knn = make_pipeline(RobustScaler(), KNeighborsClassifier())
#evaluate_classification_model_holdout([('KNN', knn)]) # .839333 instead of 0.764667

# SVC with make_pipeline
svc = make_pipeline(RobustScaler(), SVC())
#evaluate_classification_model_holdout([('SVM', svc)]) # 0.855333 instead of 0.792667

# ANN with make_pipeline
ann = make_pipeline(MinMaxScaler(), MLPClassifier())
#evaluate_classification_model_holdout([('ANN', ann)]) # 0.86 instead of 0.790667

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
'''
################ Train and test results for the model: ################
                models  accuracy_train  accuracy_test  f1_score_test  \
0   LogisticRegression        0.819250         0.8160       0.349823   
1          Naive Bayes        0.790250         0.7860       0.201493   
2                  KNN        0.875625         0.8350       0.487578   
3                  SVM        0.861750         0.8535       0.507563   
4                  ANN        0.866500         0.8575       0.573991   
5                 CART        1.000000         0.7935       0.508918   
6          BaggedTrees        0.986250         0.8445       0.544656   
7                   RF        1.000000         0.8595       0.572298   
8             AdaBoost        0.856875         0.8595       0.576169   
9                  GBM        0.874375         0.8585       0.561240   
10             XGBoost        0.956500         0.8580       0.594286   
11            LightGBM        0.913375         0.8620       0.596491   
12            CatBoost        0.910500         0.8660       0.603550   
13             NGBoost        0.861625         0.8545       0.529887   
    precision_test  recall_test  
0         0.622642     0.243243  
1         0.418605     0.132678  
2         0.662447     0.385749  
3         0.803191     0.371007  
4         0.732824     0.471744  
5         0.493088     0.525799  
6         0.673913     0.457002  
7         0.752000     0.461916  
8         0.746094     0.469287  
9         0.760504     0.444717  
10        0.709898     0.511057  
11        0.736462     0.501229  
12        0.758364     0.501229  
13        0.773585     0.402948  

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

print('F-Measure: %.3f' % f1_score(y_test, yhat_sensitive)) # 0.515
print('Accuracy: %.3f' % accuracy_score(y_test, yhat_sensitive)) # 0.718
print('Precision: %.3f' % precision_score(y_test, yhat_sensitive)) # 0.396
print('Recall: %.3f' % recall_score(y_test, yhat_sensitive)) # 0.737
print('ROC AUC Score: %.3f' % roc_auc_score(y_test, yhat_sensitive)) # 0.725


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
accuracy_score(y_test, y_pred) # 0.86
plot_classification_report_yb(rf_tuned)

# Visualization of Results --> Feature Importances
plot_feature_importances(rf_tuned)
report_results_quickly(rf_tuned)
plot_results(rf_tuned)
plot_learning_curve(rf_tuned)


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
accuracy_score(y_test, y_pred) # 0.858
plot_classification_report_yb(xgb_tuned)

# Visualization of Results --> Feature Importances
plot_feature_importances(xgb_tuned)
report_results_quickly(xgb_tuned)
plot_results(xgb_tuned)
plot_learning_curve(xgb_tuned)


# LightGBMClassifier

lgbm_model = LGBMClassifier(random_state=123456)
lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8]}

lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
lgbm_cv_model.best_params_ # {'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 500}

# Final Model
lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred) # 0.8555
plot_classification_report_yb(lgbm_tuned)

# Visualization of Results --> Feature Importances
plot_feature_importances(lgbm_tuned)
report_results_quickly(lgbm_tuned)
plot_results(lgbm_tuned)
plot_learning_curve(lgbm_tuned)


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
accuracy_score(y_test, y_pred) # 0.8635
plot_roc_auc_curve(catb_tuned)

# Visualization of Results --> Feature Importances
plot_feature_importances(catb_tuned)
report_results_quickly(catb_tuned)
plot_results(catb_tuned)
plot_learning_curve(catb_tuned)


# Comparison of tuned models

tuned_models = [('RF', rf_tuned),
                ('XGBoost', xgb_tuned),
                ('LightGBM', lgbm_tuned),
                ('CatBoost', catb_tuned)]


evaluate_classification_model_holdout(tuned_models)

