import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df_train = pd.read_csv('sample_data/housing.csv')
print(df_train)


df_train2 = df_train[['housing_median_age', 'population', 'median_house_value']]
print(df_train2.head())

train_df2_np = df_train2.to_numpy()
x_train, y_train = train_df2_np[:, :-1], train_df2_np[:, -1]

x_main, x_test, y_main, y_test=train_test_split(x_train,y_train,test_size=2500)
x_train, x_val, y_train, y_val =train_test_split(x_main,y_main, test_size=2500)
linear_model = LinearRegression().fit(x_train, y_train)
linear_train_preds = linear_model.predict(x_train)
mean_absolute_error(linear_train_preds, y_train)

