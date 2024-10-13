# Databricks notebook source
# MAGIC %md
# MAGIC ### Load raw data from csv & clean up

# COMMAND ----------

files = dbutils.fs.ls("/mnt/expt2")
display(files)

# COMMAND ----------

bank_sdf = spark.read\
  .option("header",True)\
  .option("inferenceSchema",True)\
  .csv("dbfs:/mnt/expt2/churn.csv") 

display(bank_sdf)

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog  ml_bronze;

# COMMAND ----------


databasename = "bank"
sql = f"CREATE DATABASE IF NOT EXISTS {databasename}"

spark.sql(sql)

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog ml_bronze; use database bank;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write the data frame into a delta table in the BRONZE layer

# COMMAND ----------

bank_sdf.write\
    .format('delta')\
    .mode('overwrite')\
    .saveAsTable(f"{databasename}.bank_churn")

# COMMAND ----------

import pyspark.pandas as pd
import numpy as np


# COMMAND ----------

def compute_features(spark_data_frame):

    # convert spark dataframe to pyspark dataframe
    ps_df= spark_data_frame.pandas_api()

    #drop featues which don't any value
    ps_df = ps_df.drop(['RowNumber', 'Surname'], axis=1)


    ps_df_ohe = pd.get_dummies(
        ps_df,
        columns =["Geography","Gender"],
        dtype= "int",
        drop_first=True
    )

    ps_df_ohe.columns = ps_df_ohe.columns.str.replace(r' ','', regex=True)
    ps_df_ohe.columns = ps_df_ohe.columns.str.replace(r'(','-', regex=True)
    ps_df_ohe.columns = ps_df_ohe.columns.str.replace(r')','', regex=True)

    return ps_df_ohe

# COMMAND ----------

bank_churn_final_pysdf = compute_features(bank_sdf)

display(bank_churn_final_pysdf)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fclient = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog ml_silver

# COMMAND ----------

databasename = "bank"
sql = f"CREATE DATABASE IF NOT EXISTS {databasename}"

spark.sql(sql)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC use catalog ml_silver; use database bank;

# COMMAND ----------

# MAGIC %md
# MAGIC ### create features in the silver layer 

# COMMAND ----------

bank_feature_table = fclient.create_table(
  name=f"bank_customer_features", # the name of the feature table
  primary_keys=["CustomerId"], # primary key that will be used to perform joins
  schema=bank_churn_final_pysdf.to_spark().schema,
  description="This customer level table contains one-hot encoded categorical and scaled numeric features to predict bank customer churn."
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bank_customer_features

# COMMAND ----------

# MAGIC %md
# MAGIC Go back to spark dataframe , makes it easier to work with the features and delta tables

# COMMAND ----------

bank_churn_final_pysdf\
    .to_spark()\
    .write.mode("overwrite")\
    .saveAsTable("bank_customer_features")

# COMMAND ----------


