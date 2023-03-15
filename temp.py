import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('table01.csv')



df.fillna(0, inplace=True)





X = df.drop('Total agricultural output', axis=1, inplace=False)
X = df.iloc[:, :-1]
print(X)

y = (df['Total agricultural output'])


print(X.shape)
print(y.shape)

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()


# Train a linear regression model on the training set
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

print("Coefficients:", model.coef_)

plt.scatter(y_test, y_pred)

# Plot the regression line
plt.plot(y_test, y_test, color='red')

# Set labels and title
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Linear Regression Model")


