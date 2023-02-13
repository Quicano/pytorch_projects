import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#dataset of kaggle
df_dataset = pd.read_csv('sample_data/housing.csv')
df_dataset2 = df_dataset[['housing_median_age', 'population', 'median_house_value']]

#convert to numpy Array
train_df2_np = df_dataset2.to_numpy()
x, y = train_df2_np[:, :-1], train_df2_np[:, -1]

#make train, test, validation split
x_main, x_test, y_main, y_test=train_test_split(x,y,test_size=2500)
x_train, x_val, y_train, y_val =train_test_split(x_main,y_main, test_size=2500)

#linear regression on train
linear_model = LinearRegression().fit(x_train, y_train)
linear_train_preds = linear_model.predict(x_train)

#measure the squared error
print(mean_squared_error(linear_train_preds, y_train))

#linear regression on validate
linear_model = LinearRegression().fit(x_val, y_val)
linear_val_preds = linear_model.predict(x_val)

#measure the squared error
print(mean_squared_error(linear_val_preds, y_val))

##linear regression on test
linear_model = LinearRegression().fit(x_test, y_test)
linear_test_preds = linear_model.predict(x_test)

#measure the squared error
print(mean_squared_error(linear_test_preds, y_test))

