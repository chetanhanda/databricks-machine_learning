# Databricks notebook source
# MAGIC %md
# MAGIC ### import libraries

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureLookup

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import pyspark.pandas as pd
import typing

# COMMAND ----------

# MAGIC %md
# MAGIC ### set all the params

# COMMAND ----------

#Name of the model
MODEL_NAME = "random_forest_classifier"

#This is the name for the entry in model registry
MODEL_REGISTRY_NAME = "Banking"

#Location where the MLflow experiement will be listed in user workspace
EXPERIMENT_NAME = f"Bank_Customer_Churn_Analysis"
# we have all the features backed into a Delta table so we will read directly


# COMMAND ----------

class Feature_Lookup_Input_Tuple(typing.NamedTuple):
  fature_table_name: str
  feature_list: typing.Union[typing.List[str], None] 
  lookup_key: typing.List[str]


def generate_feature_lookup(feature_mapping: typing.List[Feature_Lookup_Input_Tuple]) -> typing.List[FeatureLookup]:  
  lookups = []
  for fature_table_name, feature_list, lookup_key in feature_mapping:
    lookups.append(
          FeatureLookup(
          table_name = fature_table_name,
          feature_names = feature_list,
          lookup_key = lookup_key 
      )
    )
  return lookups

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog ml_silver;
# MAGIC select * from ml_silver.bank.bank_customer_features

# COMMAND ----------

model_feature_lookups = [FeatureLookup("ml_silver.bank.bank_customer_features","row_number")]
display(model_feature_lookups)

# COMMAND ----------

fe_client = FeatureEngineeringClient()
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    TEST_SAMPLES = 0.2

    # assemble the feature we need for training
    lookup_features = fe_client.get_feature_names(FEATURE_TABLE)
