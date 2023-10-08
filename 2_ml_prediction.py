# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Read Test Data

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM hive_metastore.default.house_prices_test

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data preprocess

# COMMAND ----------

test = _sqldf.toPandas()

test_num = test.select_dtypes(include = ['float64', 'int64'])
X_test = test_num.drop(["Id"], axis=1)

# COMMAND ----------

X_test.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load model & predict test data

# COMMAND ----------

# import mlflow
import joblib
import pandas as pd

# load
loaded_model = joblib.load("model.joblib")

# Predict on a Pandas DataFrame.
test_prediction = loaded_model.predict(pd.DataFrame(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Save result

# COMMAND ----------

import pandas as pd
import datetime

pred_df = pd.DataFrame({"id": test_num["Id"], "pred": test_prediction, "timestamp": datetime.datetime.now()})
pred_df.head()

# COMMAND ----------

# %sql
# CREATE DATABASE ml

# COMMAND ----------

# convert pandas spark
pred = spark.createDataFrame(pred_df)
pred.write.mode("overwrite").saveAsTable("ml.prediction")
