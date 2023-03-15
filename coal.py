import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error



from dataACC import df as DCdf

import requests

# api_key = '7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'



# url = 'https://api.eia.gov/v2/coal/aggregate-production/data/?frequency=annual&data[0]=average-employees&data[1]=labor-hours&data[2]=number-of-mines&data[3]=production&data[4]=productivity&start=2010&end=2020&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'
# response = requests.get(url)
# js = response.json()
# data = pd.json_normalize(js)


with open('coal.txt', 'r') as f:
    data_str = f.read()

data = json.loads(data_str)

df = pd.DataFrame(data)

df = pd.concat([df, DCdf])

df = df.drop('coalRankDescription', axis=1, inplace=False)
df = df.drop('average-employees-units', axis=1, inplace=False)
df = df.drop('labor-hours-units', axis=1, inplace=False)
df = df.drop('number-of-mines-units', axis=1, inplace=False)
df = df.drop('production-units', axis=1, inplace=False)
df = df.drop('mineTypeDescription', axis=1, inplace=False)
df = df.drop('productivity-units', axis=1, inplace=False)
df = df.drop('stateRegionDescription', axis=1, inplace=False)
df = df.drop('stateRegionId', axis=1, inplace=False)
df = df.drop('coalRankId', axis=1, inplace=False)
df = df.drop('mineTypeId', axis=1, inplace=False)


# encoding = pd.get_dummies(df['stateRegionId'])
# new_df = pd.concat([df, encoding], axis=1)
# df = new_df.drop('stateRegionId', axis=1)

# encoding = pd.get_dummies(df['mineTypeId'])
# new_df = pd.concat([df, encoding], axis=1)
# df = new_df.drop('mineTypeId', axis=1)



#df = df.drop('productivity', axis=1, inplace=False)




lst = []

for i in df['period']:

    lst.append(i)

lst = [*set(lst)]

# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['labor-hours']])
#     df['labor-hours'] = imputer.transform(df[['labor-hours']])


# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['average-employees']])
#     df['average-employees'] = imputer.transform(df[['average-employees']])

# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['production']])
#     df['production'] = imputer.transform(df[['production']])

# for i in lst:


#     subset = df[df['period'] == i]
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(subset[['number-of-mines']])
#     df['number-of-mines'] = imputer.transform(df[['number-of-mines']])


df = df[df['average-employees'] != 0]
df = df[df['average-employees'].notna()]

df = df[df['labor-hours'] != 0]
df = df[df['labor-hours'].notna()]

df = df[df['number-of-mines'] != 0]
df = df[df['number-of-mines'].notna()]

df = df[df['production'] != 0]
df = df[df['production'].notna()]

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



df.to_csv('coal.csv')



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









# encoding = pd.get_dummies(df['stateRegionId'])
# new_df = pd.concat([df, encoding], axis=1)
# new_df = new_df.drop('stateRegionId', axis=1)

# encoding = pd.get_dummies(df['coalRankId'])
# new_df = pd.concat([new_df, encoding], axis=1)
# new_df = new_df.drop('coalRankId', axis=1)

# encoding = pd.get_dummies(df['mineTypeId'])
# new_df = pd.concat([new_df, encoding], axis=1)
# df = new_df.drop('mineTypeId', axis=1)








# subset_df = df[df['period'] == 2020].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# df.to_csv('coal.csv')
# subset_df = df[df['period'] == 2019].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2018].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2017].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2016].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2015].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2014].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2013].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2012].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)

# subset_df = df[df['period'] == 2011].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)
# subset_df = df[df['period'] == 2010].copy()  
# imputer = SimpleImputer(strategy='mean')
# imputer.fit(subset_df[['labor-hours']])
# subset_df.loc[:, 'labor-hours'] = imputer.transform(subset_df[['labor-hours']])
# df.update(subset_df)

















# X = df.drop('number-of-mines', axis=1, inplace=False)

# y = (df['number-of-mines'])
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









# scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
# mse_scores = -scores
# print("Mean Squared Error:", np.mean(mse_scores))

# print("Coefficients:", model.coef_)










# plt.scatter(y_test, y_pred)

# # Plot the regression line
# plt.plot(y_test, y_test, color='red')

# # Set labels and title
# plt.xlabel("Actual values")
# plt.ylabel("Predicted values")
# plt.title("Linear Regression Model")
# #plt.show()




# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)