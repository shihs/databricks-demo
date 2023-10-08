# Databricks notebook source
# MAGIC %md
# MAGIC ## Read data from bucket

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("spark_test").getOrCreate()

# COMMAND ----------

path = "gs://databricks-demo-data/charts.csv"
df = spark.read.csv(path, sep=',', inferSchema=True, header=True)


# COMMAND ----------

df.toPandas().head(3)

# COMMAND ----------

df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 使用 fs

# COMMAND ----------

# MAGIC %fs

# COMMAND ----------

# MAGIC %fs ls dbfs:/

# COMMAND ----------

# MAGIC %fs ls dbfs:/user/hive/warehouse/

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/data

# COMMAND ----------

# MAGIC %fs ls dbfs:/data

# COMMAND ----------

# MAGIC %fs head dbfs:/data/Advertising.csv

# COMMAND ----------

path = "dbfs:/data/Advertising.csv"
df = spark.read.csv(path, sep=',', inferSchema=True, header=True)
df.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## SQL
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SHOW TABLES;

# COMMAND ----------

_sqldf.show(10)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM samples.nyctaxi.trips

# COMMAND ----------

_sqldf.show(10)

# COMMAND ----------


