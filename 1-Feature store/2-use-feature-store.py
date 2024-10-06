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

inference_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement")).orderBy("real_time_measurement")
display(inference_df)

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name, lookup_key)]

    training_set = fs.create_training_set(inference_df,model_feature_lookups,label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    X = training_pd.drop(["quality"], axis=1)
    y= training_pd["quality"]
    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    return X_train,X_test,y_train,y_test,training_set
    



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
        rf = RandomForestRegressor(max_depth=3,n_estimators=20, random_state=42)
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse",mean_squared_error(y_test,y_pred))
        mlflow.log_metric("test_r2_score",r2_score(y_test,y_pred))

        fs.log_model(
            model = rf,
            artifact_path = "wine_quality_predicition",
            flavor = mlflow.sklearn,
            training_set = training_set,
            registered_model_name = "wine_model",
        )


train_model(X_train,y_train,X_test,y_test,training_set,fs)                          

