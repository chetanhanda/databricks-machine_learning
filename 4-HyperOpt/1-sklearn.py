# Databricks notebook source
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC using california housing dataset

# COMMAND ----------

X, y = fetch_california_housing(return_X_y=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Normalizing the features so that they don't cause any bias because of their high values. e.g house price is in milltions and rooms are on <10
# MAGIC Using sklearn's standard scaler

# COMMAND ----------

# confirming/checking how much the imbalance is if we don't do anything 
X.mean(axis=0)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# COMMAND ----------

# now checking the new values , compared to the previous ones
X.mean(axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC Converting y(target) into a category. All the house prices are integers which is a continous value. 
# MAGIC If the value of the house is above a  median it's expensive , make it 1. If it's below median make it 0

# COMMAND ----------

y_discrete = np.where(y < np.median(y), 0, 1)

# COMMAND ----------

def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    else:
        return 0
    accuracy = cross_val_score(clf, X, y_discrete).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}
