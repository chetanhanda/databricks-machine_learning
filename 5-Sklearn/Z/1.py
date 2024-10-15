# Databricks notebook source
# MAGIC %md
# MAGIC ### load the csv

# COMMAND ----------

import pyspark.pandas as pd

# COMMAND ----------

files = dbutils.fs.ls("/mnt/expt-sklearn")
display(files)

# COMMAND ----------


heart_disease_pys_Df = spark.read\
    .option("header",True)\
    .option("inferenceSchema",True)\
    .csv("dbfs:/mnt/expt-sklearn/heart-disease.csv") 

display(heart_disease_pys_Df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### convert spark dataframe to pyspark panda dataframe

# COMMAND ----------

heart_disease_pysPd_Df = heart_disease_pys_Df.toPandas()
display(heart_disease_pysPd_Df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### get the features loaded

# COMMAND ----------

# X = features matrix
X = heart_disease_pysPd_Df.drop('target', axis=1)

# y = labels
y = heart_disease_pysPd_Df['target']

# COMMAND ----------

# MAGIC %md
# MAGIC ### choose right models and params

# COMMAND ----------

# using a classification ml model and hyper parameters
from sklearn.ensemble import RandomForestClassifier

# default hyperparams
clf = RandomForestClassifier()

# see what the default is 
clf.get_params()



# COMMAND ----------

# MAGIC %md
# MAGIC ### fit the model to the data 

# COMMAND ----------

# get the  data split up 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# COMMAND ----------


