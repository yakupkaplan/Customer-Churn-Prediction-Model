# CUSTOMER CHURN PREDICTION

# EXPLORATORY DATA ANALYSIS, DATA PREPROCESSING AND FEATURE ENGINEERING

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
    Surname,CreditScore, Geography (Country: France, Spain, Germany), Gender (Female / Male), Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    Exited: Churn or not? (0 = No, 1 = Yes)

Steps for Exploratory Data Analysis, Data Visualization, Data Preprocessing and Feature Engineering:
    ??? SMOTE --> Imbalanced data
    - GENERAL / GENERAL OVERVIEW / GENERAL PICTURE
    - NUMERICAL VARIABLE ANALYSIS
        describe with quantiles to see whether there are extraordinary values or not
        Basic visualization by using histograms
    - TARGET ANALYSIS
        Target analysis according to categorical variables --> target_summary_with_cats()
        Target analysis according to numerical variables --> target_summary_with_nums()
    - ANALYSIS OF NUMERCIAL VARIABLES IN COMPARISON WITH EACH OTHER
        scatterplot
        lmplot
        correlation
    - Outlier Analysis
    - Missing Values Analysis
    - New Features Creation
    - Label and One Hot Encoding
    - Standardization
    - Saving the Dataset
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

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import preprocessing

import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%config InlineBackend.figure_format = 'retina'

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);


# Load the dataset
churn = pd.read_csv("datasets/churn.csv", index_col=0)
df = churn.copy()
df.head()


## GENERAL VIEW

df.head()
df.shape # (10000, 13)
df.info()
df.columns
df.index
df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# Check for missing values
df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False)

# Drop 'CustomerId' column, because we do not need it for our analysis.
df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# Let's define numerical and categorical features
df.columns

num_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in ['Exited']]
print('Number of Numerical Variables : ', len(num_cols), '-->', num_cols) # ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

cat_cols = [col for col in df.columns if df[col].dtype == "O"] # ['Surname', 'Geography', 'Gender']
print('Number of Categorical Variables : ', len(cat_cols), '-->', cat_cols)

# After analysis, we redefined categorical and numerical variables
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
print('Number of Numerical Variables : ', len(num_cols), '-->', num_cols)

cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
print('Number of Categorical Variables : ', len(cat_cols), '-->', cat_cols)

# numbers of unique classes for each cat_cols
df[cat_cols].nunique()


## CATEGORICAL VARIABLES ANALYSIS


# Function that catches categorical variables, prints the distribution and ratio of unique values and finally creates a countplot
def cats_summary1(data, target):
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10]
    for col in cat_names:
        print(pd.DataFrame({col: data[col].value_counts(),
                            "Ratio": 100 * data[col].value_counts() / len(data),
                            "TARGET_MEAN": data.groupby(col)[target].mean()}), end="\n\n\n")
        sns.countplot(x=col, data=data)
        plt.show()


cats_summary1(df, 'Exited')


# Function takes dataframe, categorical columns and if required number of classes (default value = 10).
# Then, it prints the distribution and ratio of unique values for the variables, which have less than 10 (number of classes) unique values.
# Afterwards, it reports categorical variables it described, how many numerical variables, but seem categorical we have and finally report these variables.
def cats_summary2(data, categorical_cols, target, number_of_classes=10):
    var_count = 0  # reporting how many categorical variables are there?
    vars_more_classes = []  # save the variables that have classes more than a number that we determined

    for var in data.columns:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # select according to number of classes
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data),
                                    "TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cats_summary2(df, cat_cols, 'Exited')


## NUMERICAL VARIABLES ANALYSIS

# Plot histograms for the dataset
df.hist(bins=20, figsize=(15, 15), color='r');
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


hist_for_nums(df, num_cols)


## TARGET ANALYSIS

df["Exited"].value_counts() # --> We see, that here we are dealing with imbalanced data.

# See how many 0 and 1 values in the dataset and if there is imbalance
sns.countplot(x='Exited', data=df);
plt.show()

# Look at the mean and meadian for each variable groupped by Outcome
for col in num_cols:
    print(df.groupby("Exited").agg({col: ['count', "mean", "median"]}), '\n')


## ANALYSIS OF NUMERCIAL VARIABLES IN COMPARISON WITH EACH OTHER

# Show the scatterplots for each variable and add the dimension for Outcome, so we can differentiate between classes.
sns.pairplot(df, hue='Exited');
plt.show()

# Show the correlation matrix
plt.subplots(figsize=(15, 12))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, cmap='coolwarm', annot=True, square=True);
plt.show()


## OUTLIER ANALYSIS


# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
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


has_outliers(df, num_cols)
df['NumOfProducts'].value_counts()
# NumOfProducts : 60 --> These are the values with 4 Products. We do not remove this whole class. It can have valuable information for us.


# # Assign outliers thresholds values for all the numerical variables
# for col in df.columns:
#     replace_with_thresholds_with_lambda(df, col)


## MISSING VALUES ANALYSIS

# Check for missing values
df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False) # There are no missing values in the dataset.


## FEATURE CREATION

# Create


## LABEL AND ONE HOT ENCODING

# Catch numerical variables
cat_cols


# Define a function to apply one hot encoding to categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=False, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, cat_cols)
df.head()
len(new_cols_ohe)

df.info()

# Export the dataset for later use by modeling
df.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\churn_prepared.csv', index=False)

'''
# STANDARDIZATION

df.head()

# Catch numerical variables
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in ["Outcome"]]
len(num_cols)

# MinMaxScaler

df_minmax_scaled = df.copy()

from sklearn.preprocessing import MinMaxScaler
transformer = MinMaxScaler()
df_minmax_scaled[num_cols] = transformer.fit_transform(df_minmax_scaled[num_cols])  # default value is between 0 and 1

df_minmax_scaled[num_cols].describe().T
len(num_cols)

# StandardScaler

df_std_scaled = df.copy()

from sklearn.preprocessing import StandardScaler
transformer = StandardScaler()
df_std_scaled[num_cols] = transformer.fit_transform(df_std_scaled[num_cols])

df_std_scaled[num_cols].describe().T
len(num_cols)

# RobustScaler

df_robust_scaled = df.copy()

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler()
df_robust_scaled[num_cols] = transformer.fit_transform(df_robust_scaled[num_cols])

df_robust_scaled[num_cols].describe().T
len(num_cols)

# Check before modeling for missing values and outliers in the dataset
# df.isnull().sum().sum()
# has_outliers(df, num_cols)
df_minmax_scaled.isnull().sum().sum()
has_outliers(df_minmax_scaled, num_cols)

df_std_scaled.isnull().sum().sum()
has_outliers(df_std_scaled, num_cols)

df_robust_scaled.isnull().sum().sum()
has_outliers(df_robust_scaled, num_cols)

# Last look at the dataset
df.head()
df.info()

# Export the dataset for later use by modeling
#df.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\diabetes_prepared.csv')
df_minmax_scaled.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\churn_prepared_minmaxscaled.csv', index=False)
df_std_scaled.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\churn_prepared_stdscaled.csv', index=False)
df_robust_scaled.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\churn_prepared_robustscaled.csv', index=False)
'''


