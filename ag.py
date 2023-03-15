import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


df = pd.read_csv('table01.csv')



df.fillna(0, inplace=True)


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
    df.loc[df['Year'] ==period, 'GDP'] = gdp


# X = df.drop('Total agricultural output', axis=1, inplace=False)

# y = (df['Total agricultural output'])


X = df.drop('GDP', axis=1, inplace=False)

y = (df['GDP'])



model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Train a linear regression model on the training set
model.fit(X_train, y_train)

y_pred = model.predict(X_test)





mse = mean_squared_error(y_test, y_pred)

#print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred)

#print("Mean Absolute Error:", mae)

rmse = np.sqrt(mse)
error_percentage = rmse / (y_test.max() - y_test.min()) * 100
#print("Percentage Error:", error_percentage)
