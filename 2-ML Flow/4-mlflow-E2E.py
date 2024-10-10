# Databricks notebook source
import pandas as pd

# COMMAND ----------

df= dbutils.fs.ls("/mnt/something/")
display(df)

# COMMAND ----------

white_wine =  spark.read.csv("/mnt/something/winequality-white.csv",sep=";", header=True)
red_wine = spark.read.csv("/mnt/something/winequality-red.csv",sep=";", header=True)

display(white_wine)

# COMMAND ----------

def format_column_names(df):
    """Rename columns """
    renamed_df = df
    for col in df.columns:                
        renamed_df = renamed_df.withColumnRenamed(col, col.replace(" ", "_").lower())                                          
    return renamed_df

# COMMAND ----------

# concat the two datasets
concat_df = white_wine.union(red_wine)
display(concat_df)

# COMMAND ----------

renamed_df = format_column_names(concat_df) 
display(renamed_df) 

# COMMAND ----------


