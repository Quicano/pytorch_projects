import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



df_train = pd.read_csv('sample_data/california_housing_test.csv')
df_train.head()

df_test = pd.read_csv('sample_data/california_housing_test.csv')
df_test.head()

df_train2 = df_train[['housing_median_age', 'population', 'median_house_value']]
df_train2.head()


df_test2 = df_test[['housing_median_age', 'population', 'median_house_value']]
df_test2.head()
