# Databricks notebook source
import pandas as pd

# COMMAND ----------

df= dbutils.fs.ls("/mnt/something/")
display(df)

# COMMAND ----------

white_wine = pd.read_csv("/dbfs/mnt/something/winequality-white.csv",sep=";")
red_wine = pd.read_csv("/dbfs/mnt/something/winequality-red.csv",sep=";")

# COMMAND ----------


