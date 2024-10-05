# Databricks notebook source
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup


# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import monotonically_increasing_id, expr,rand
import uuid

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets'))
display(dbutils.fs.ls('dbfs:/databricks-datasets/wine-quality/'))

# COMMAND ----------

raw_data = spark.read.load("dbfs:/databricks-datasets/wine-quality/winequality-red.csv", format="csv", sep=";", header="true", inferSchema="True")

# COMMAND ----------

display(raw_data)

# COMMAND ----------


