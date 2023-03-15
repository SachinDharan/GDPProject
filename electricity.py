import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error



import requests

# api_key = '7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'



# url = 'https://api.eia.gov/v2/coal/aggregate-production/data/?frequency=annual&data[0]=average-employees&data[1]=labor-hours&data[2]=number-of-mines&data[3]=production&data[4]=productivity&start=2010&end=2020&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'
# response = requests.get(url)
# js = response.json()
# data = pd.json_normalize(js)





# with open('elec.txt', 'r') as f:
#     data_str = f.read()

# data = json.loads(data_str)

# df = pd.DataFrame(data)




# df = df.drop('stateDescription', axis=1, inplace=False)
# df = df.drop('sectorName', axis=1, inplace=False)
# df = df.drop('customers-units', axis=1, inplace=False)
# df = df.drop('price-units', axis=1, inplace=False)
# df = df.drop('revenue-units', axis=1, inplace=False)
# df = df.drop('sales-units', axis=1, inplace=False)



# # encoding = pd.get_dummies(df['stateid'])
# # new_df = pd.concat([df, encoding], axis=1)
# # new_df = new_df.drop('stateid', axis=1)

# encoding = pd.get_dummies(df['sectorid'])
# new_df = pd.concat([df, encoding], axis=1)
# df = new_df.drop('sectorid', axis=1)




# df = df.drop('stateid', axis=1)
# #df = df.drop('sectorid', axis=1)
# df['period'] = pd.to_datetime(df['period'])
# df['year'] = df['period'].dt.year
# df['month'] = df['period'].dt.month
# df = df.drop('period', axis=1)
# df = df.drop('price', axis=1, inplace=False)
# df = df.drop('COM', axis=1, inplace=False)
# df = df.drop('OTH', axis=1, inplace=False)
# df = df.drop('IND', axis=1, inplace=False)
# df = df.drop('TRA', axis=1, inplace=False)
# df = df.drop('ALL', axis=1, inplace=False)
# df = df.drop('RES', axis=1, inplace=False)








# plt.subplots(figsize=(20,15))
# sns.heatmap(
#     df.corr(method ='pearson'), 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     annot=True, annot_kws={'size': 10},
#     square=True, xticklabels=df.columns,
#     yticklabels=df.columns)


# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# #plt.show()




# #lets drop some more columns with low correlation


# df = df.drop('sales', axis=1, inplace=False)




# # df.fillna(0, inplace=True)

# imputer = SimpleImputer(strategy='mean')

# # Fit the imputer to the data
# imputer.fit(df)

# new_df = imputer.transform(df)

# df = pd.DataFrame(new_df, columns=df.columns)

# df.to_csv('elec.csv')





# X = df.drop('customers', axis=1, inplace=False)
# print(X)

# y = (df['customers'])





# model = LinearRegression()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)



# mse = mean_squared_error(y_test, y_pred)

# print("Mean Squared Error:", mse)



with open('altElec.txt', 'r') as f:
    data_str = f.read()

data = json.loads(data_str)

df = pd.DataFrame(data)




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
    df.loc[df['period'] ==period, 'GDP'] = gdp


df.loc[df['period'] ==2022, 'GDP'] = 23944.58716



df = df.drop('customers', axis=1, inplace=False)



# # df = df.drop('stateDescription', axis=1, inplace=False)
# # df = df.drop('sectorName', axis=1, inplace=False)
df = df.drop('customers-units', axis=1, inplace=False)
df = df.drop('price-units', axis=1, inplace=False)
df = df.drop('revenue-units', axis=1, inplace=False)
df = df.drop('sales-units', axis=1, inplace=False)
df = df.drop('stateDescription', axis=1, inplace=False)
df = df.drop('sectorName', axis=1, inplace=False)
# df = df.drop('stateid', axis=1, inplace=False)
# df = df.drop('sectorid', axis=1, inplace=False)


df = df[df['revenue'] != 0]
df = df[df['revenue'].notna()]
df = df[df['price'] != 0]
df = df[df['price'].notna()]
df = df[df['sales'] != 0]
df = df[df['sales'].notna()]


df.to_csv('elec.csv')

# lst = []

# for i in df['period']:

#     lst.append(i)

# lst = [*set(lst)]




encoding = pd.get_dummies(df['stateid'])
new_df = pd.concat([df, encoding], axis=1)
df = new_df.drop('stateid', axis=1)

encoding = pd.get_dummies(df['sectorid'])
new_df = pd.concat([df, encoding], axis=1)
df = new_df.drop('sectorid', axis=1)




# lst = []

# for i in df['period']:

#     lst.append(i)

# lst = [*set(lst)]

# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['revenue']])
#     df['revenue'] = imputer.transform(df[['revenue']])

# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['sales']])
#     df['sales'] = imputer.transform(df[['sales']])

# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['price']])
#     df['price'] = imputer.transform(df[['price']])







# plt.subplots(figsize=(20,15))
# sns.heatmap(
#     df.corr(method ='pearson'), 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     annot=True, annot_kws={'size': 10},
#     square=True, xticklabels=df.columns,
#     yticklabels=df.columns)


# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# #plt.show()





# # # df.fillna(0, inplace=True)

# # imputer = SimpleImputer(strategy='mean')

# # # Fit the imputer to the data
# # imputer.fit(df)

# # new_df = imputer.transform(df)

# # df = pd.DataFrame(new_df, columns=df.columns)

# # df.to_csv('elec.csv')





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

