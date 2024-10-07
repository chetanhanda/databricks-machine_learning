# Databricks notebook source
raw_new_df = spark.sql("select * from global_temp.df_temp")

df_reference = raw_new_df.drop("quality")
display(df_reference)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fs = FeatureEngineeringClient()

# Correct usage with keyword arguments
predictions_df = fs.score_batch(
    model_uri="models:/wine_model/latest", 
    df=df_reference
)

# Correctly display the predictions
display(predictions_df.select("wine_id", "prediction"))

# COMMAND ----------


