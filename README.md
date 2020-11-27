# Customer-Churn-Prediction-Model

__Problem__

- Can you develop a machine learning model that can predict customers who will leave the company (customer churn)?
- The aim is to predict whether a bank's customers leave the bank or not.
- The situation that defines the customer churn is the customer closing his bank account.

__Data Set Story__

It consists of 10000 observations and 12 variables.
Independent variables contain information about customers.
The dependent variable expresses the customer churn situation.

__Variables__

Surname,CreditScore, Geography (Country: France, Spain, Germany), Gender (Female / Male), Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
Exited: Churn or not? (0 = No, 1 = Yes)

__Steps for Exploratory Data Analysis, Data Visualization, Data Preprocessing and Feature Engineering__

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

__There is also exercise with PySpark. You may see the differences between pandas and pyspark operations.__
    
    
