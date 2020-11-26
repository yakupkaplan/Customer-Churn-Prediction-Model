# CUSTOMER CHURN PREDICTION

'''
In this project, like before customer churn will be predicted. Here, PySpark will be used.

Some references:
    https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
    https://stackoverflow.com/questions/60772315/how-to-evaluate-a-classifier-with-apache-spark-2-4-5-and-pyspark-python
    https://gist.github.com/AlessandroChecco/c930a8b868342fa34b23a1f282dc3e88
'''


# EXPLORATORY DATA ANALYSIS AND DATA PREPROCESSING

# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import pyspark dependencies
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.classification import GBTClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.conf import SparkConf
from pyspark import SparkContext
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import findspark
findspark.init("C:\spark")

spark = SparkSession.builder.master("local").appName("churn_prediction").config("spark.executer.memory", "16gb").getOrCreate()
sc = spark.sparkContext
sc
# sc.stop()


# Load the dataset
spark_df = spark.read.csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\churn.csv", header=True, inferSchema=True)
spark_df.printSchema()
type(spark_df)


### EXPLORATORY DATA ANALYSIS

## GENERAL VIEW

spark_df.head()
print("Shape of the dataset", (spark_df.count(), len(spark_df.columns)))
spark_df.printSchema()
pd.DataFrame(spark_df.take(5), columns=spark_df.columns).transpose()
spark_df.columns
spark_df.schema
spark_df.dtypes

# Summary statistics for numerical variables
numeric_features = [t[0] for t in spark_df.dtypes if t[1] == 'int']
spark_df.select(numeric_features).describe().toPandas().transpose()

# Check for missing values
from pyspark.sql.functions import when, count, col
spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

# Lower all feature names for simplicity
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns]) # lower all feature names
spark_df.show(5)

# After analysis, we redefined categorical and numerical variables
num_cols = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary']
print('Number of Numerical Variables : ', len(num_cols), '-->', num_cols)

cat_cols = ['geography', 'gender', 'hascrcard', 'isactivemember']
print('Number of Categorical Variables : ', len(cat_cols), '-->', cat_cols)

# Unique classes for each cat_cols
for col in cat_cols:
    spark_df.select(col).distinct().show()


## Data Understanding

# value_counts
spark_df.groupby("exited").count().show()

# Numerical Variables Summary
for col in [col.lower() for col in num_cols]:
    spark_df.groupby("exited").agg({col: "mean"}).show()

# Categorical Variables Summary
for col in [col.lower() for col in cat_cols]:
    spark_df.crosstab("exited", col).show()

# Statistical sumamry for selected columns
spark_df.select("age", "creditscore", "tenure", "balance", "numofproducts", 'hascrcard', 'isactivemember', 'estimatedsalary', "exited").describe().toPandas().transpose()

# Filtering
spark_df.filter(spark_df.age > 70).count()

# Groupby for target variable
spark_df.groupby("exited").count().show()

# See the means with respect to target variable
for col in ["age", "creditscore", "tenure", "balance", "numofproducts", 'hascrcard', 'isactivemember', 'estimatedsalary']:
    spark_df.groupby("exited").agg({col: "mean"}).show()


### DATA PREPROCESSING & FEATURE ENGINEERING

# Rename index column
spark_df = spark_df.withColumnRenamed("rownumber", "index") # rename index column
spark_df.show(5)
spark_df.columns

# Drop the columns that we do not need for our Model
spark_df = spark_df.drop('index', 'customerid', 'surname')
spark_df.show(5)


##  Feature Creation

# Create a feature that show age categories  18-30 --> 1, 30-40 --> 2, 40-50 --> 3, 50-60 --> 4, 60-92 --> 5
from pyspark.ml.feature import Bucketizer
spark_df.select('age').describe().toPandas().transpose()
bucketizer = Bucketizer(splits=[0, 30, 40, 50, 60, 92], inputCol="age", outputCol="age_ranges")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df.show(5)
#cat_cols.append('age_ranges')
# See the results for the new feature
spark_df.select("exited", "age_ranges").describe().show()
spark_df.groupby("age_ranges").count().show()
spark_df.groupby("age_ranges").agg({'exited': "mean"}).show()


# Alternative way for age_ranges

# spark_df.withColumn('age_ranges', F.when(spark_df['age'] < 30, 1).when(30<spark_df['age']<40, 2).when(40<spark_df['age']<50, 3).when(50<spark_df['age']<60, 4).otherwise(5))

# Funtion for age_ranges
def categorizer(age):
    if age < 30:
        return 1
    elif age < 40:
        return 2
    elif age < 50:
        return 3
    elif age < 60:
        return 4
    else:
        return 5


func_udf = udf(categorizer, IntegerType())
spark_df = spark_df.withColumn('age_ranges', func_udf(spark_df['age']))
spark_df.show(5)
cat_cols.append('age_ranges')
# See the results for the new feature
spark_df.select("exited", "age_ranges").describe().show()
spark_df.groupby("age_ranges").count().show()
spark_df.groupby("age_ranges").agg({'exited': "mean"}).show()


# Create a feature that shows credit score ranges
spark_df.select('creditscore').describe().toPandas().transpose()
bucketizer = Bucketizer(splits=[300, 500, 601, 661, 781, 851], inputCol="creditscore", outputCol="creditscore_ranges")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df.show(5)
cat_cols.append('creditscore_ranges')
# See the results for the new feature
spark_df.select("exited", "creditscore_ranges").describe().show()
spark_df.groupby("creditscore_ranges").count().show()
spark_df.groupby("creditscore_ranges").agg({'exited': "mean"}).show()


# Create a feature that shows Tenure/NumOfProducts
spark_df = spark_df.withColumn('tenure/numofproducts', spark_df.tenure/spark_df.numofproducts)
spark_df.show(5)
num_cols.append('tenure/numofproducts')
# See the results for the new feature
spark_df.select("exited", "tenure/numofproducts").describe().show()
spark_df.groupby("exited").agg({'tenure/numofproducts': "mean"}).show()


# Create a new feature called age/tenure
spark_df = spark_df.withColumn('tenure/age', spark_df.tenure/spark_df.age)
spark_df.show(5)
num_cols.append('tenure/age')
# See the results for the new feature
spark_df.select("exited", "tenure/age").describe().show()
spark_df.groupby("exited").agg({'tenure/age': "mean"}).show()


# Create a feature that shows EstimatedSalary/Age
spark_df = spark_df.withColumn('estimatedsalary/age', spark_df.estimatedsalary/spark_df.age)
spark_df.show(5)
num_cols.append('estimatedsalary/age')
# See the results for the new feature
spark_df.select("exited", "estimatedsalary/age").describe().show()
spark_df.groupby("exited").agg({'estimatedsalary/age': "mean"}).show()


# Create a feature that shows Balance/ESalary
spark_df = spark_df.withColumn('balance/estimatedsalary', spark_df.balance/spark_df.estimatedsalary)
spark_df.show(5)
num_cols.append('balance/estimatedsalary')
# See the results for the new feature
spark_df.select("exited", "balance/estimatedsalary").describe().show()
spark_df.groupby("exited").agg({'balance/estimatedsalary': "mean"}).show()


# Create a feature that shows ESalary/Tenure
spark_df = spark_df.withColumn('estimatedsalary/tenure', spark_df.estimatedsalary/spark_df.tenure)
spark_df.show(5)
num_cols.append('estimatedsalary/tenure')
# See the results for the new feature
spark_df.select("exited", "estimatedsalary/tenure").describe().show()
spark_df.groupby("exited").agg({'estimatedsalary/tenure': "mean"}).show()


# All of those below 405 are churned (20 values), they remained on the edge like outlier, we did not throw them, we created a new variable.

# Funtion for Filtering
def smallerthan405(creditscore):
    if creditscore < 405:
        return 1
    else:
        return 0


func_udf = udf(smallerthan405, IntegerType())
spark_df = spark_df.withColumn('smallerthan405', func_udf(spark_df['creditscore']))
cat_cols.append('smallerthan405')
# See the results for the new feature
spark_df.select("exited", "smallerthan405").describe().show()
spark_df.groupby("smallerthan405").count().show()
spark_df.groupby("exited").agg({'smallerthan405': "mean"}).show()

# spark_df.select(when(spark_df['creditscore'] < 405, 1).otherwise(0).alias("smallerthan405")).collect()


# Create a feature that shows whther 'Balance' < 0 or not.

# Funtion for Filtering
def hasbalance(balance):
    if balance > 0:
        return 1
    else:
        return 0


func_udf = udf(hasbalance, IntegerType())
spark_df = spark_df.withColumn('hasbalance', func_udf(spark_df['balance']))
cat_cols.append('hasbalance')
# See the results for the new feature
spark_df.select("exited", "hasbalance").describe().show()
spark_df.groupby("hasbalance").count().show()
spark_df.groupby("exited").agg({'hasbalance': "mean"}).show()


# # Drop Features that we will not ues anymore
# spark_df.drop(['creditscore', 'balance'], axis=1, inplace=True)


## Missing Values

# Check for missing values
from pyspark.sql.functions import when, count, col
spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

# Drop missing values
spark_df = spark_df.dropna()


## Setting Label, Features and One-Hot or Label Encoding

# Remember categorical and numerical variables
print(cat_cols)
print(num_cols)


# Define dependent variable/target
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
indexed = stringIndexer.fit(spark_df).transform(spark_df)

# Change the type of label to int.
spark_df = indexed.withColumn("label", indexed["label"].cast("integer"))

# Let's look at the dataset
spark_df.printSchema()
spark_df.show()

# Label and One Hot Encoding

# geograpyhy
indexer = StringIndexer(inputCol="geography", outputCol="geography_categoryIndex")
indexed1 = indexer.fit(spark_df).transform(spark_df)
indexed1 = indexed1.withColumn("geography_categoryindex", indexed1["geography_categoryindex"].cast("integer"))

# gender
indexer = StringIndexer(inputCol="gender", outputCol="gender_categoryindex")
indexed1 = indexer.fit(indexed1).transform(indexed1)
indexed1 = indexed1.withColumn("gender_categoryindex", indexed1["gender_categoryindex"].cast("integer"))

indexed1.show(5)

# One Hot Encoding

indexed1.select("geography_categoryindex").distinct().count()

encoder = OneHotEncoder(inputCols=["geography_categoryindex"], outputCols=["geography_categoryindex_vec"])
indexed2 = encoder.fit(indexed1).transform(indexed1)

# Look at the dataset and see the changes
indexed2.show()
indexed2.dtypes
indexed2.columns

columns = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary',
           'age_ranges', 'creditscore_ranges', 'tenure/numofproducts', 'tenure/age', 'estimatedsalary/age', 'balance/estimatedsalary',
           'estimatedsalary/tenure', 'smallerthan405', 'hasbalance', 'geography_categoryindex', 'gender_categoryindex', 'geography_categoryindex_vec']

# Vectorize independent variables.
va = VectorAssembler(inputCols=columns, outputCol="features")
va_df = va.transform(indexed2)

va_df.show()

# Final dataframe
final_df = va_df.select("features", "label")
final_df.show(5)

# StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
final_scaled_df = scaler.fit(final_df).transform(final_df)


# Split the dataset into test and train sets.
train_df, test_df = final_scaled_df.randomSplit([0.7, 0.3], seed=2018)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))


### MODELING

# Define evaluator
evaluator = BinaryClassificationEvaluator()

# Logistic Regression

logreg = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
logreg_model = logreg.fit(train_df)

# Make predictions
y_pred = logreg_model.transform(test_df)
y_pred.show()

# Calculate accuracy
ac = y_pred.select("label", "prediction")
ac.show(5)
ac.filter(ac.label == ac.prediction).count() / ac.count() # 0.8189775910364145


# Show other metrics

# Make predicitons
predictionAndTarget = logreg_model.transform(test_df).select("label", "prediction")

# Create both evaluators
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')

# Get metrics
acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
roc_auc = evaluator.evaluate(predictionAndTarget)

# Return the results
msg = "LogisticRegression: accuracy: %f, roc_auc: %f, f1_score: %f, weightedPrecision: %f, weightedRecall: %f" % (acc, roc_auc, f1, weightedPrecision, weightedRecall)
print(msg)


# Decision Tree Classifier

dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=3)
dt_model = dt.fit(train_df)

# Make predictions
y_pred = dt_model.transform(test_df)
y_pred.show()

# Calculate accuracy
ac = y_pred.select("label", "prediction")
ac.show(5)
ac.filter(ac.label == ac.prediction).count() / ac.count()

# Get metrics
acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictionAndTarget)

# Return the results
msg = "DecisionTreeClassifier: accuracy: %f, roc_auc: %f, f1_score: %f, weightedPrecision: %f, weightedRecall: %f" % (acc, auc, f1, weightedPrecision, weightedRecall)
print(msg)


# Random Forest Classifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rf_model = rf.fit(train_df)

# Make predictions
y_pred = rf_model.transform(test_df)
y_pred.show()

# Calculate accuracy
ac = y_pred.select("label", "prediction")
ac.show(5)
ac.filter(ac.label == ac.prediction).count() / ac.count() # 0.8452702702702702

# Get metrics
acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictionAndTarget)

# Return the results
msg = "RandomForestClassifier: accuracy: %f, roc_auc: %f, f1_score: %f, weightedPrecision: %f, weightedRecall: %f" % (acc, auc, f1, weightedPrecision, weightedRecall)
print(msg)


# GBM

gbm = GBTClassifier(maxIter=10, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)

# Make predictions
y_pred = gbm_model.transform(test_df)
y_pred.show()

# Calculate accuracy
ac = y_pred.select("label", "prediction")
ac.show(5)
ac.filter(ac.label == ac.prediction).count() / ac.count() # 0.8577702702702703

# Get metrics
acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictionAndTarget)

# Return the results
msg = "GBTClassifier: accuracy: %f, roc_auc: %f, f1_score: %f, weightedPrecision: %f, weightedRecall: %f" % (acc, auc, f1, weightedPrecision, weightedRecall)
print(msg)


## Model Tuning

evaluator = BinaryClassificationEvaluator()

paramGrid = (ParamGridBuilder()
             .addGrid(gbm.maxDepth, [2, 4, 6])
             .addGrid(gbm.maxBins, [20, 30])
             .addGrid(gbm.maxIter, [10, 20])
             .build())

cv = CrossValidator(estimator=gbm, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
cv_model = cv.fit(train_df)

y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
evaluator.evaluate(y_pred)


# Stop Spark session
sc.stop()
