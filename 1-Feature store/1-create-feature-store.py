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

raw_df = spark.read.load("dbfs:/databricks-datasets/wine-quality/winequality-red.csv", format="csv", sep=";", header="true", inferSchema="True")

# COMMAND ----------

display(raw_df)

# COMMAND ----------

def addIdColumn(dataframe, id_column_name):
    """Add id column to the dataframe"""
    original_columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + original_columns]

def cleanColumnNames(df):
    """Rename columns """
    renamed_df = df
    for col in df.columns:        
        renamed_df = renamed_df.withColumnRenamed(col, col.replace(" ", "_").lower())                                          
    return renamed_df
    



# COMMAND ----------

renamed_raw_df = cleanColumnNames(raw_df)
display(renamed_raw_df)


# COMMAND ----------

renamed_raw_df = addIdColumn(renamed_raw_df, "wine_id")
display(renamed_raw_df)

# COMMAND ----------

trimmed_df = renamed_raw_df.drop("quality")
display(trimmed_df) 

# COMMAND ----------


