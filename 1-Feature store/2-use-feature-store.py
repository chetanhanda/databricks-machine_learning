# Databricks notebook source
from pyspark.sql.functions import rand
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

import pandas as pd
from pyspark.sql.functions import monotonically_increasing_id, expr,rand
import uuid

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow.sklearn

df = spark.sql("select * from global_temp.df_temp")
display(df)

# COMMAND ----------

# contains some features having newer values which will not be present in the feature store
# wine_id will be the primary key based on which this raw dataset will be joined to the dataframe
raw_new_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement")).orderBy("real_time_measurement")
display(raw_new_df)

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

def load_data(table_name, lookup_key):
    
    #get the features and the cooresponding lookup key based on which the raw dataset will be joined to the feature store
    model_feature_lookups = [FeatureLookup(table_name, lookup_key)]

    #training set = aggregate of the features from the dataframe and the feature store 
    training_set = fs.create_training_set(
        df=raw_new_df, # this is the raw data used for training, can contain some newer data which may not be present in the feature store
        feature_lookups=model_feature_lookups, # all the features to include in the training set, thes will be joined into the inference_df
        label="quality", # target column for supervised learning, can be set to "none" for unsupervised learning 
        exclude_columns="wine_id" # exclude the features which are not going to add any value
        )
    
    training_pd = training_set.load_df().toPandas() # convert the aggregate dataframe to Pandaas dataframe 

    X = training_pd.drop(["quality"], axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, training_set

# COMMAND ----------

X_train,X_test,y_train,y_test,training_set = load_data("ml.wine_db.wine_0b4044","wine_id")

# COMMAND ----------

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLFLOW

# COMMAND ----------

from  mlflow.tracking.client import MlflowClient

client  = MlflowClient()

try:
    client.delete_registered_model("wine_model")
except:
    None    


# COMMAND ----------


mlflow.sklearn.autolog(log_models=False)

def train_model(X_train,y_train, X_test,y_test,training_set,fs):
    with mlflow.start_run() as run:
        rf = RandomForestRegressor(max_depth=3,n_estimators=20, random_state=42) # set the hyperparameters
        rf.fit(X_train,y_train) # try different hyperparameters and features
        y_pred = rf.predict(X_test)  # compare the predicted values with the actual values in X_test

        mlflow.log_metric("test_mse",mean_squared_error(y_test,y_pred)) # log the mean square error
        mlflow.log_metric("test_r2_score",r2_score(y_test,y_pred)) # log the mean r2 score

        fs.log_model(
            model = rf,
            artifact_path = "wine_quality_predicition",
            flavor = mlflow.sklearn,
            training_set = training_set,
            registered_model_name = "wine_model",
        )


train_model(X_train,y_train,X_test,y_test,training_set,fs)                          


# COMMAND ----------


