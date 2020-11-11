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

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning

import pickle
import warnings
warnings.simplefilter(action="ignore")
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
churn = pd.read_csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\churn.csv", index_col=0)
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

# Drop 'CustomerId' and 'Surname' column, because we do not need them for our model.
df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# Let's define numerical and categorical features
df.columns

num_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in ['Exited']]
print('Number of Numerical Variables : ', len(num_cols), '-->', num_cols) # ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

cat_cols = [col for col in df.columns if df[col].dtype == "O"] # ['Geography', 'Gender']
print('Number of Categorical Variables : ', len(cat_cols), '-->', cat_cols)

# After analysis, we redefined categorical and numerical variables
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
print('Number of Numerical Variables : ', len(num_cols), '-->', num_cols)

cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
print('Number of Categorical Variables : ', len(cat_cols), '-->', cat_cols)

# numbers of unique classes for each cat_cols
df[cat_cols].nunique()


## CATEGORICAL VARIABLES ANALYSIS

# Show summary for categorical variables
fn.cats_summary1(df, 'Exited')


## NUMERICAL VARIABLES ANALYSIS

# Plot histograms for the dataset
df.hist(bins=20, figsize=(15, 15), color='r');
plt.show()

# Show histogtams for numerical variables
fn.hist_for_nums(df, num_cols)


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

# See, if we have any outlier in the dataset
fn.has_outliers(df, num_cols)

# Assign outliers thresholds values for all the numerical variables
for col in num_cols:
    fn.replace_with_thresholds_with_lambda(df, col)

# Check for outlier , again.
fn.has_outliers(df, num_cols)


## MISSING VALUES ANALYSIS

# Check for missing values
df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False) # There are no missing values in the dataset.


## FEATURE CREATION

# Create a feature that show age categories  18-30 --> 1, 30-40 --> 2, 40-50 --> 3, 50-60 --> 4, 60-92 --> 5
df['Age'].describe()
df['AgeRanges'] = pd.cut(x=df['Age'], bins=[0, 30, 40, 50, 60, 92], labels=[1, 2, 3, 4, 5])
cat_cols.append('AgeRanges')
# See the results for the new feature
df.groupby(["Exited", "AgeRanges"]).describe()
df[['AgeRanges']].value_counts()
df.groupby(["AgeRanges"]).agg({"Exited": [np.mean, np.size]}) # Super!

# Create a feature that shows credit score ranges
df['CreditScore'].describe()
df['CreditScoreRanges'] = pd.cut(x=df['CreditScore'], bins=[300, 500, 601, 661, 781, 851], labels=[1, 2, 3, 4, 5])
cat_cols.append('CreditScoreRanges')
# See the results for the new feature
df.groupby(["Exited", "CreditScoreRanges"]).describe()
df[['CreditScoreRanges']].value_counts()
df.groupby(["CreditScoreRanges"]).agg({"Exited": [np.mean, np.size]}) # Super!

# Create a feature that shows Tenure/NumOfProducts
df["Tenure/NumOfProducts"] = df["Tenure"]/df["NumOfProducts"]
num_cols.append('Tenure/NumOfProducts')
# See the results for the new feature
sns.boxplot(x='Exited', y='Tenure/NumOfProducts', data=df)
plt.show()

# Create a feature that shows EstimatedSalary/Age
df["ESalary/Age"] = df["EstimatedSalary"]/(df["Age"])
num_cols.append('ESalary/Age')
# See the results for the new feature
sns.boxplot(x='Exited', y='ESalary/Age', data=df)
plt.show()

# Create a feature that shows Tenure/Age
df["Tenure/Age"] = df["Tenure"]/(df["Age"])
num_cols.append('Tenure/Age')
# See the results for the new feature
sns.boxplot(x='Exited', y='Tenure/Age', data=df)
plt.show()

# Create a feature that shows Balance/ESalary
df["Balance/ESalary"] = df["Balance"]/(df["EstimatedSalary"])
num_cols.append('Balance/ESalary')
# See the results for the new feature
sns.boxplot(x='Exited', y='Balance/ESalary', data=df)
plt.show()

# Create a feature that shows ESalary/Tenure
df["ESalary/Tenure"] = df["EstimatedSalary"]/(df["Tenure"]+1)
num_cols.append('ESalary/Tenure')
# See the results for the new feature
sns.boxplot(x='Exited', y='ESalary/Tenure', data=df)
plt.show()

# # Create a feature that shows ESalary/Tenure
# df["ESalary/CreditScoreranges"] = df["EstimatedSalary"]/(df["CreditScoreRanges"])
# # See the results for the new feature
# sns.boxplot(x='Exited', y='ESalary/CreditScoreranges', data=df)
# plt.show()

# All of those below 405 are churned (20 values), they remained on the edge like outlier, we did not throw them, we created a new variable.
df["SmallerThan405"] = df['CreditScore'].apply(lambda x: 1 if x < 405 else 0)
cat_cols.append('SmallerThan405')
df["SmallerThan405"].value_counts()
# See the results for the new feature
df.groupby(["SmallerThan405"]).agg({"Exited": [np.mean, np.size]}) # Super!

# Create a feature that shows whther 'Balance' < 0 or not.
df['HasBalance'] = df['Balance'].apply(lambda x: 1 if x > 0 else 0)
cat_cols.append('HasBalance')
df['HasBalance'].value_counts()
# See the results for the new feature
df.groupby(["HasBalance"]).agg({"Exited": [np.mean, np.size]}) # Super!

# Drop Features that we will not ues anymore
df.drop(['CreditScore', 'Balance'], axis=1, inplace=True)


## LABEL AND ONE HOT ENCODING

# Catch numerical variables
cat_cols

df, new_cols_ohe = fn.one_hot_encoder(df, cat_cols)
df.head()
len(new_cols_ohe)

df.info()

# STANDARDIZATION

# Standardization will be implemented in the modeling phase for distance based algoritms.


# Export the dataset for later use by modeling
df.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_churn_prediction\churn_prepared.csv', index=False)




