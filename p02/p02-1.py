import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# loading dataset
df = pd.read_csv('data/dataset.csv')
print("\n")
print("Data loaded successfully")
print('Shape:', df.shape)
print(df.head())

# Visualizing the data
df.plot(x = 'Investimento', y = 'Retorno', style = 'o')
plt.title('Investiment vs Profit')
plt.xlabel('Investiment')
plt.ylabel('Profit')
plt.savefig('images/p02-1-1.png')
plt.show()

# Preparing the data

X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

# Splitting the data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

# Adjusting the shape of the train data
X_train = X_train.reshape(-1,1).astype(np.float32)


# Linear Regression Model
model = LinearRegression()

# Training the model
model.fit(X_train, Y_train)
print("\n")
print('Model trained successfully')

#Printing the coefficients B0 and B1
print("\n")
print('B1 (coef_) :', model.coef_)
print('B0 (intercept_) :', model.intercept_)

# Plotting the regression line
# Y = B0 + B1*X
regression_line = model.coef_ * X + model.intercept_
plt.scatter(X, Y)
plt.title('Investiment vs Profit')
plt.xlabel('Investiment')
plt.ylabel('Profit')
plt.plot(X, regression_line, color='red')
plt.savefig('images/p02-1-regressionline.png')
plt.show()


# Making predictions
y_pred = model.predict(X_test)

# Comparing the actual and predicted values
df_values = pd.DataFrame({'Real Value': Y_test, 'Predicted Value': y_pred})
print("\n")
print(df_values)

#Plot 
fig, ax = plt.subplots()
index = np.arange(len(X_test))
bar_width = 0.35
actual = plt.bar(index, df_values['Real Value'], bar_width, label='Real Value')
predicted = plt.bar(index + bar_width, df_values['Predicted Value'], bar_width, label='Predicted Value')
plt.xlabel('Investiment')
plt.ylabel('Profit')
plt.title('Investiment vs Profit')
plt.xticks(index+bar_width, X_test)
plt.legend()
plt.savefig('images/p02-1-actual-predictaded.png')
plt.show()

# Evaluating the model
print("\n")
print('MAE (Mean Absolute Error):', mean_absolute_error(Y_test, y_pred))
print('MSE (Mean Squared Error):', mean_squared_error(Y_test, y_pred))
print('RMSE (Root Mean Squared Error):', np.sqrt(mean_squared_error(Y_test, y_pred)))
print('R2 Score:', r2_score(Y_test, y_pred))


# Predicting the profit for a new investment

print("\n")
input_env = input('Enter the investment value: ')
input_inv = float(input_env)
inv = np.array([input_inv])
inv = inv.reshape(-1,1)

profit = model.predict(inv)
print("\n")
print('Investiment: ', input_inv)
print('Predicted Profit = {:.4}'.format(profit[0]))
print("\n")
