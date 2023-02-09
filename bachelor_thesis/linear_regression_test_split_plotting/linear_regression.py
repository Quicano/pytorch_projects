import pandas as pd

df_train = pd.read_csv('sample_data/california_housing_test.csv')
df_train.head()

df_test = pd.read_csv('sample_data/california_housing_test.csv')
df_test.head()

df_train2 = df_train[['housing_median_age', 'population', 'median_house_value']]
df_train2.head()


df_test2 = df_test[['housing_median_age', 'population', 'median_house_value']]
df_test2.head()

train_df2_np = df_train2.to_numpy()
X_train, y_train = train_df2_np[:, -1], train_df2_np[:, -1]
X_train.shape, y_train.shape
