# Databricks notebook source
# MAGIC %md
# MAGIC ## Import

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from numpy import savetxt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the data

# COMMAND ----------

db = load_diabetes()
X= db.data
y= db.target

X_train, X_test , y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check the distribution after splitting

# COMMAND ----------

print("Number of observations in X_train are :" , len(X_train))
print("Number of observations in X_test are :" , len(X_test))
print("Number of observations in y_train are :" , len(y_train))
print("Number of observations in y_test are :" , len(y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check version of mlflow
# MAGIC

# COMMAND ----------

print(mlflow.version.VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regresssion with RandomforestRegressor

# COMMAND ----------

# Enable autolog  
mlflow.sklearn.autolog()

# since autolog is enabled , all maodel parameters, model score, fitted model is automatically logged
with mlflow.start_run() as run:

    # hyper parameters
    n_estimators = 100
    max_depth = 6
    max_features = 3

    # create and train model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    rf.fit(X_train, y_train)

    #use the model to try and make predictions
    y_pred = rf.predict(X_test)
    
    #log the model's params
    mlflow.log_param("n_estimators_log", n_estimators)
    mlflow.log_param("max_depth_log", max_depth)
    mlflow.log_param("max_features_log", max_features)

    #Define the mse 
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    # log the model created by this run
    mlflow.sklearn.log_model(rf, "random forest model")

    # save the table of predicted values
    savetxt("predictions.csv",y_pred,delimiter=",")

    # log the saved table as an artifact
    mlflow.log_artifact("predictions.csv")

    # convert to panda dataframe for display
    df = pd.DataFrame(data=y_pred-y_test)

    # create a plot
    plt.plot(df)
    plt.xlabel("Observation")
    plt.xlabel("Residual")
    plt.xlabel("Residuals")

    # Save the plot as an artifact
    plt.savefig("residuals_plot.png")
    mlflow.log_artifact("residuals_plot.png")
    



# COMMAND ----------

mlflow.end_run()
