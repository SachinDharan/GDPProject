import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from workingData import df as wdDF
from sklearn.metrics import mean_absolute_error


import requests

# api_key = '7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'



# url = 'https://api.eia.gov/v2/coal/aggregate-production/data/?frequency=annual&data[0]=average-employees&data[1]=labor-hours&data[2]=number-of-mines&data[3]=production&data[4]=productivity&start=2010&end=2020&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'
# response = requests.get(url)
# js = response.json()
# data = pd.json_normalize(js)





with open('natGas.txt', 'r') as f:
    data_str = f.read()

data = json.loads(data_str)

df = pd.DataFrame(data)

df.to_csv('natGas.csv')

df = pd.concat([df, wdDF])




df = df.drop('units', axis=1, inplace=False)
df = df.drop('area-name', axis=1, inplace=False)
df = df.drop('product', axis=1, inplace=False)
df = df.drop('product-name', axis=1, inplace=False)
df = df.drop('process-name', axis=1, inplace=False)
df = df.drop('series-description', axis=1, inplace=False)
df = df.drop('series', axis=1, inplace=False)
df = df.drop('duoarea', axis=1, inplace=False)
df = df.drop('process', axis=1, inplace=False)


lst = []

for i in df['period']:

    lst.append(i)

lst = [*set(lst)]


df['value'] = df['value'].replace(0, None)


# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['value']])
#     df['value'] = imputer.transform(df[['value']])





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

df = df[df['value'] != 0]
df = df[df['value'].notna()]


df.to_csv('natGas.csv')




plt.subplots(figsize=(20,15))
sns.heatmap(
    df.corr(method ='pearson'), 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    annot=True, annot_kws={'size': 10},
    square=True, xticklabels=df.columns,
    yticklabels=df.columns)


plt.xticks(rotation=90)
plt.yticks(rotation=0)
#plt.show()

lst = [1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990]
for i in lst:
    df = df[df['period'] != i]



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

