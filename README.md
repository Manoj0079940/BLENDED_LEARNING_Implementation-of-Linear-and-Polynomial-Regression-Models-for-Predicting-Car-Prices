# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection:
Import essential libraries like pandas, numpy, sklearn, matplotlib, and seaborn. Load the dataset using pandas.read_csv().

2.Data Preprocessing:
Address any missing values in the dataset. Select key features for training the models. Split the dataset into training and testing sets with train_test_split().

3.Linear Regression:
Initialize the Linear Regression model from sklearn. Train the model on the training data using .fit(). Make predictions on the test data using .predict(). Evaluate model performance with metrics such as Mean Squared Error (MSE) and the R² score.

4.Polynomial Regression:
Use PolynomialFeatures from sklearn to create polynomial features. Fit a Linear Regression model to the transformed polynomial features. Make predictions and evaluate performance similar to the linear regression model.

5.Visualization:
Plot the regression lines for both Linear and Polynomial models. Visualize residuals to assess model performance.
## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data

df = pd.read_csv('encoded_car_data.csv')
print(df.head())
# Select features & target
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression (with scaling)

linear_model = Pipeline([
('scaler', StandardScaler()),
('model', LinearRegression())])

linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

#2. Polynomial Regression (degree=2)

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scalar', StandardScaler()),
    ('model', LinearRegression())])

poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

# Evaluate Models

print("Linear Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_linear):.2f")
print(f"R2: {r2_score(y_test, y_pred_linear):.2f}")

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2: {r2_score(y_test, y_pred_poly):.2f}")

# Plot actual vs predicted

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial (degree=2)', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
![Screenshot 2025-04-18 100208](https://github.com/user-attachments/assets/c0a27070-96ef-4157-8afc-1c70dcfdda29)
![Screenshot 2025-04-18 100157](https://github.com/user-attachments/assets/0038abc4-394d-4eec-b90d-cb3df9d175a6)
![Screenshot 2025-04-18 100134](https://github.com/user-attachments/assets/239c9042-4f7f-43ac-b485-e6ddc1c0658b)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
