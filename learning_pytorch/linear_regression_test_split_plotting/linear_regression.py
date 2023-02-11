import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('sample_data/california_housing_test.csv')
df_train.head()

df_test = pd.read_csv('sample_data/california_housing_test.csv')
df_test.head()

df_train2 = df_train[['housing_median_age', 'population', 'median_house_value']]
df_train2.head()


df_test2 = df_test[['housing_median_age', 'population', 'median_house_value']]
df_test2.head()

train_df2_np = df_train2.to_numpy()
X_train, y_train = train_df2_np[:, :-1], train_df2_np[:, -1]
X_train.shape, y_train.shape


linear_model = LinearRegression().fit(X_train, y_train)
linear_train_preds = linear_model.predict(X_train)
mean_absolute_error(linear_train_preds, y_train)

test_df2_np = df_test2.to_numpy()
X_test, y_test = test_df2_np[:, :-1], test_df2_np[:, -1]
X_test.shape, y_test.shape

linear_test_preds = linear_model.predict(X_test)
mean_absolute_error(linear_test_preds, y_test)

rf = RandomForestRegressor(n_estimators=10, max_depth=5).fit(X_train, y_train)
rf_train_preds = rf.predict(X_train)
rf_test_preds = rf.predict(X_test)
mean_absolute_error(rf_train_preds, y_train), mean_absolute_error(rf_test_preds, y_test)

X_val, X_hold, y_val, y_hold = train_test_split(X_test, y_test, test_size=0.5)
X_val.shape, X_hold.shape, y_val.shape, y_hold.shape

linear_val_preds = linear_model.predict(X_val)
mean_absolute_error(y_train, linear_train_preds), mean_absolute_error(y_val, linear_val_preds)

rf_val_preds = rf.predict(X_val)
mean_absolute_error(y_train, rf_train_preds), mean_absolute_error(y_val, rf_val_preds)

rf_hold_preds = rf.predict(X_hold)
mean_absolute_error(y_hold, rf_hold_preds)