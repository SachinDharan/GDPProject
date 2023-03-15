import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingRegressor
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from coal import model as coalModel
from coal import y_pred as coalPred
from electricity import model as elecModel
from electricity import y_pred as elecPred
from Manufacturing import model as manModel
from Manufacturing import y_pred as manPred
from NatGas import model as NGModel
from NatGas import y_pred as ngPred
from ag import model as agModel
from ag import y_pred as agPred
from sklearn.metrics import r2_score



df = pd.read_csv('GDP.csv')

X = df['period'].values.reshape(-1, 1) # reshape to 2D array with one column
y = df[' GDP ( Billions of US $)'].values # reshape to 2D array with one column

modLst = []

modLst.append(('coal', coalModel))
modLst.append(('electricity', elecModel))
modLst.append(('Manufacturing', manModel))
modLst.append(('NatGas', NGModel))
modLst.append(('Agriculture', agModel))


model = StackingRegressor(estimators=modLst, final_estimator=LinearRegression())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)

#print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)

rmse = np.sqrt(mse)
error_percentage = rmse / (y_test.max() - y_test.min()) * 100
print("Percentage Error:", error_percentage)
r2 = r2_score(y_test, y_pred)

# Print R-squared score
print("R-squared score:", r2)
#print(model.coef_)


# print(coalPred.shape)
# print(elecPred.shape)

# average = np.mean([coalPred, elecPred, manPred, agPred, ngPred], axis=0)
# mse = mean_squared_error(y_test, average)

