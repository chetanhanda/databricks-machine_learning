# Databricks notebook source
# MAGIC %md
# MAGIC ### load the csv

# COMMAND ----------

import pyspark.pandas as pd
import numpy as np

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

clf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### try making a prediction

# COMMAND ----------

# make a prediction with X_test
X_test
y_preds = clf.predict(X_test)
y_preds

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the model

# COMMAND ----------

# model score on the training data
clf.score(X_train,y_train)

# COMMAND ----------

# model score on the test data
clf.score(X_test,y_test)

# COMMAND ----------

from sklearn.metrics import classification_report,confusion_matrix  ,accuracy_score

print(classification_report(y_test,y_preds))

# COMMAND ----------

confusion_matrix(y_test,y_preds)

# COMMAND ----------

accuracy_score(y_test,y_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ### try different params to improve model

# COMMAND ----------

np.random.seed(42)
for i in range(10,100,5):
  print(f"Trying model with {i} estimators..")
  clf = RandomForestClassifier(n_estimators=i).fit(X_train,y_train)
  print(f"Model accuracy on test set : {clf.score(X_test,y_test)*100:2f}%")
  print("")

# COMMAND ----------

# MAGIC %md
# MAGIC ### choose the best value of the estimator and train model with that value of the hyperparameter i.e 20

# COMMAND ----------

clf = RandomForestClassifier(n_estimators=20).fit(X_train,y_train)
print(f"Model accuracy on test set : {clf.score(X_test,y_test)*100:2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### save the model

# COMMAND ----------

import pickle

pickle.dump(clf,open("random_forest_model_1.pkl","wb"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### load the model

# COMMAND ----------

loaded_model= pickle.load(open("random_forest_model_1.pkl","rb"))
loaded_model.score(X_test,y_test)

# COMMAND ----------


