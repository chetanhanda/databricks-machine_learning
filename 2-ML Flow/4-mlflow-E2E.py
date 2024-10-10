# Databricks notebook source
#import pandas as pd
import pyspark.pandas as pd


# COMMAND ----------

df= dbutils.fs.ls("/mnt/something/")
display(df)

# COMMAND ----------

white_wine =  pd.read_csv("/mnt/something/winequality-white.csv",sep=";")
red_wine = pd.read_csv("/mnt/something/winequality-red.csv",sep=";")

red_wine['is_red']=1
white_wine['is_red']=0

display(red_wine)
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
concat_df = pd.concat([white_wine, red_wine], ignore_index=True)
display(concat_df)

# COMMAND ----------

# renamed_df = format_column_names(concat_df) 
renamed_df = concat_df.rename(columns=lambda x: x.replace(" ", "_").lower())
display(renamed_df)

# COMMAND ----------

display(renamed_df.describe)

# COMMAND ----------

renamed_df.corr

# COMMAND ----------

display(renamed_df.head())

# COMMAND ----------

renamed_df.dtypes

# COMMAND ----------

renamed_df.describe()

# COMMAND ----------


