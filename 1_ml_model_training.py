# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Read Training Data From DB

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM hive_metastore.default.house_prices_train

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data fields
# MAGIC Here's a brief version of what you'll find in the data description file.
# MAGIC
# MAGIC - 🌟 **SalePrice** - the property's sale price in dollars. This is the target variable that you're trying to predict.
# MAGIC - **MSSubClass**: The building class
# MAGIC - **MSZoning**: The general zoning classification
# MAGIC - **LotFrontage**: Linear feet of street connected to property
# MAGIC - **LotArea**: Lot size in square feet
# MAGIC - **Street**: Type of road access
# MAGIC - **Alley**: Type of alley access
# MAGIC - **LotShape**: General shape of property
# MAGIC - **LandContour**: Flatness of the property
# MAGIC - **Utilities**: Type of utilities available
# MAGIC - **LotConfig**: Lot configuration
# MAGIC - **LandSlope**: Slope of property
# MAGIC - **Neighborhood**: Physical locations within Ames city limits
# MAGIC - **Condition1**: Proximity to main road or railroad
# MAGIC - **Condition2**: Proximity to main road or railroad (if a second is present)
# MAGIC - **BldgType**: Type of dwelling
# MAGIC - **HouseStyle**: Style of dwelling
# MAGIC - **OverallQual**: Overall material and finish quality
# MAGIC - **OverallCond**: Overall condition rating
# MAGIC - **YearBuilt**: Original construction date
# MAGIC - **YearRemodAdd**: Remodel date
# MAGIC - **RoofStyle**: Type of roof
# MAGIC - **RoofMatl**: Roof material
# MAGIC - **Exterior1st**: Exterior covering on house
# MAGIC - **Exterior2nd**: Exterior covering on house (if more than one material)
# MAGIC - **MasVnrType**: Masonry veneer type
# MAGIC - **MasVnrArea**: Masonry veneer area in square feet
# MAGIC - **ExterQual**: Exterior material quality
# MAGIC - **ExterCond**: Present condition of the material on the exterior
# MAGIC - **Foundation**: Type of foundation
# MAGIC - **BsmtQual**: Height of the basement
# MAGIC - **BsmtCond**: General condition of the basement
# MAGIC - **BsmtExposure**: Walkout or garden level basement walls
# MAGIC - **BsmtFinType1**: Quality of basement finished area
# MAGIC - **BsmtFinSF1**: Type 1 finished square feet
# MAGIC - **BsmtFinType2**: Quality of second finished area (if present)
# MAGIC - **BsmtFinSF2**: Type 2 finished square feet
# MAGIC - **BsmtUnfSF**: Unfinished square feet of basement area
# MAGIC - **TotalBsmtSF**: Total square feet of basement area
# MAGIC - **Heating**: Type of heating
# MAGIC - **HeatingQC**: Heating quality and condition
# MAGIC - **CentralAir**: Central air conditioning
# MAGIC - **Electrical**: Electrical system
# MAGIC - **1stFlrSF**: First Floor square feet
# MAGIC - **2ndFlrSF**: Second floor square feet
# MAGIC - **LowQualFinSF**: Low quality finished square feet (all floors)
# MAGIC - **GrLivArea**: Above grade (ground) living area square feet
# MAGIC - **BsmtFullBath**: Basement full bathrooms
# MAGIC - **BsmtHalfBath**: Basement half bathrooms
# MAGIC - **FullBath**: Full bathrooms above grade
# MAGIC - **HalfBath**: Half baths above grade
# MAGIC - **Bedroom**: Number of bedrooms above basement level
# MAGIC - **Kitchen**: Number of kitchens
# MAGIC - **KitchenQual**: Kitchen quality
# MAGIC - **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms)
# MAGIC - **Functional**: Home functionality rating
# MAGIC - **Fireplaces**: Number of fireplaces
# MAGIC - **FireplaceQu**: Fireplace quality
# MAGIC - **GarageType**: Garage location
# MAGIC - **GarageYrBlt**: Year garage was built
# MAGIC - **GarageFinish**: Interior finish of the garage
# MAGIC - **GarageCars**: Size of garage in car capacity
# MAGIC - **GarageArea**: Size of garage in square feet
# MAGIC - **GarageQual**: Garage quality
# MAGIC - **GarageCond**: Garage condition
# MAGIC - **PavedDrive**: Paved driveway
# MAGIC - **WoodDeckSF**: Wood deck area in square feet
# MAGIC - **OpenPorchSF**: Open porch area in square feet
# MAGIC - **EnclosedPorch**: Enclosed porch area in square feet
# MAGIC - **3SsnPorch**: Three season porch area in square feet
# MAGIC - **ScreenPorch**: Screen porch area in square feet
# MAGIC - **PoolArea**: Pool area in square feet
# MAGIC - **PoolQC**: Pool quality
# MAGIC - **Fence**: Fence quality
# MAGIC - **MiscFeature**: Miscellaneous feature not covered in other categories
# MAGIC - **MiscVal**: $Value of miscellaneous feature
# MAGIC - **MoSold**: Month Sold
# MAGIC - **YrSold**: Year Sold
# MAGIC - **SaleType**: Type of sale
# MAGIC - **SaleCondition**: Condition of sale

# COMMAND ----------

# type(_sqldf)
df = _sqldf.toPandas()
df.head(3)

# COMMAND ----------

df.shape

# COMMAND ----------

df.info()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### House Price Distribution

# COMMAND ----------

print(df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

# https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf

# COMMAND ----------

df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

# COMMAND ----------

df_num.shape

# COMMAND ----------

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prepare Dateset

# COMMAND ----------

df_num = df_num[["Id", "SalePrice", 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]

# COMMAND ----------

df_num.shape

# COMMAND ----------

import numpy as np

def split_dataset(dataset, test_ratio=0.30):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(df_num)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

# COMMAND ----------

import mlflow
# import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
 
from numpy import savetxt
 
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_diabetes
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import joblib


# COMMAND ----------

mlflow.__version__

# COMMAND ----------

X_train = train_ds_pd.drop(["Id", "SalePrice"], axis=1)
y_train = train_ds_pd["SalePrice"]

X_valid = valid_ds_pd.drop(["Id", "SalePrice"], axis=1)
y_valid = valid_ds_pd["SalePrice"]

# COMMAND ----------

X_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model

# COMMAND ----------

# mlflow.sklearn.autolog()
username = "mcshihs@gmail.com"
mlflow.set_experiment(f"/Users/{username}/test_exp")
with mlflow.start_run():

    # Set the model parameters. 
    n_estimators = 300
    max_depth = 10
    max_features = 20

    # Create and train model.
    rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf.fit(X_train, y_train)
    # save
    joblib.dump(rf, "model.joblib")

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_valid)

    # Log the model parameters used for this run.
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_feat", max_features)
    
    # Define a metric to use to evaluate the model.
    mse = mean_squared_error(y_valid, predictions)
    # Log the value of the metric from this run.
    mlflow.log_metric("mse", mse)

    # Log the model created by this run.
    mlflow.sklearn.log_model(rf, "random-forest-model") 

    # Save the table of predicted values
    savetxt('predictions.csv', predictions, delimiter=',')

    # Log the saved table as an artifact
    mlflow.log_artifact("predictions.csv")

    # Convert the residuals to a pandas dataframe to take advantage of graphics capabilities
    df = pd.DataFrame(data = predictions - y_valid)

    # Create a plot of residuals
    plt.plot(df)
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("Residuals")

    # Save the plot and log it as an artifact
    plt.savefig("residuals_plot.png")
    mlflow.log_artifact("residuals_plot.png") 

# https://docs.gcp.databricks.com/_extras/notebooks/source/mlflow/mlflow-quick-start-python.html
# https://docs.gcp.databricks.com/_extras/notebooks/source/mlflow/mlflow-logging-api-quick-start-python.html


# COMMAND ----------

!ls -la

# COMMAND ----------

pd.read_csv("predictions.csv", header=None).head()

# COMMAND ----------

# !rm predictions.csv & rm model.joblib & rm residuals_plot.png & ls
