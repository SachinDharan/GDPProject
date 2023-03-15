import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error




df = pd.read_excel('Manufacturing.xlsx')


df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%y')

df['year'] = df['DATE'].dt.year
df['month'] = df['DATE'].dt.month
df['day'] = df['DATE'].dt.day
df['dayofweek'] = df['DATE'].dt.dayofweek

df.drop('DATE', axis=1, inplace=True)

tempDF = pd.read_csv('gdp.csv')

tempCol = tempDF[' GDP ( Billions of US $)']


lst = []

for index, row in tempDF.iterrows():
    period = row.loc['period']
    gdp = row.loc[' GDP ( Billions of US $)']
    vals = (int(period), gdp)
    lst.append(vals)



for i in lst:
    # access the individual elements of the tuple using indexing
    period = i[0]
    gdp = i[1]
    df.loc[df['year'] ==period, 'GDP'] = gdp


df.loc[df['year'] ==2022, 'GDP'] = 23944.58716
df.loc[df['year'] ==2023, 'GDP'] = 23944.58716
df = df.drop('Capital Goods', axis=1, inplace=False)
df.fillna(0, inplace=True)


df.to_csv('man.csv')







# plt.subplots(figsize=(20,15))
# sns.heatmap(
#     df.corr(method ='pearson'), 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     annot=True, annot_kws={'size': 3},
#     square=True, xticklabels=df.columns,
#     yticklabels=df.columns)


# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# #plt.show()





# # print("X_train shape:", X_train.shape)
# # print("X_test shape:", X_test.shape)
# # print("y_train shape:", y_train.shape)
# # print("y_test shape:", y_test.shape)
# # print()




X = df.drop('GDP', axis=1, inplace=False)

y = (df['GDP'])





model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)

#print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred)

#print("Mean Absolute Error:", mae)

rmse = np.sqrt(mse)
error_percentage = rmse / (y_test.max() - y_test.min()) * 100
#print("Percentage Error:", error_percentage)

