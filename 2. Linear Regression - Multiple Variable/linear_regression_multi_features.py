### STEP - 1
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os


### STEP - 2
# Importing the dataset
# Define the file path
file_path = 'datasets/homeprices_multiple_variables.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Preview the data
print("Dataset Preview:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nDataset Statistics:")
print(df.describe())


### STEP - 3
# Data Visualization - Exploring relationships between features and price
plt.figure(figsize=(15, 4))

# Plot 1: Area vs Price
plt.subplot(1, 3, 1)
plt.scatter(df['area'], df['price'], color='blue', marker='o', s=100, alpha=0.6)
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (US$)')
plt.title('Area vs Price')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Bedrooms vs Price
plt.subplot(1, 3, 2)
plt.scatter(df['bedrooms'], df['price'], color='green', marker='o', s=100, alpha=0.6)
plt.xlabel('Bedrooms')
plt.ylabel('Price (US$)')
plt.title('Bedrooms vs Price')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 3: Age vs Price
plt.subplot(1, 3, 3)
plt.scatter(df['age'], df['price'], color='red', marker='o', s=100, alpha=0.6)
plt.xlabel('Age (years)')
plt.ylabel('Price (US$)')
plt.title('Age vs Price')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


### STEP - 4
# Preparing the data
# X contains all independent variables (features)
# y contains the dependent variable (target)
X = df[['area', 'bedrooms', 'age']]
y = df['price']

print("\nFeatures (X):")
print(X)
print("\nTarget (y):")
print(y)


### STEP - 5
# Training the Linear Regression model
model = LinearRegression()
model.fit(X, y)

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE")
print("="*50)

# Display model parameters
print("\nModel Coefficients:")
print(f"  Area coefficient: {model.coef_[0]:.2f} (Price change per sq ft)")
print(f"  Bedrooms coefficient: {model.coef_[1]:.2f} (Price change per bedroom)")
print(f"  Age coefficient: {model.coef_[2]:.2f} (Price change per year of age)")
print(f"  Intercept: {model.intercept_:.2f}")

# Model evaluation on training data
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\nModel Performance Metrics:")
print(f"  R² Score: {r2:.4f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse:.2f}")


### STEP - 6
# Making predictions on test data
print("\n" + "="*50)
print("MAKING PREDICTIONS ON TEST DATA")
print("="*50)

# Load the test dataset
test_file_path = 'datasets/testing.csv'
test_homes = pd.read_csv(test_file_path)

print("\nTest Dataset:")
print(test_homes)

# Generate predictions on test data
predictions = model.predict(test_homes)
test_homes['predicted_price'] = predictions

print("\nPredicted Prices for Test Data:")
print(test_homes)


### STEP - 7
# Visualizing predictions vs actual data
fig = plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted for Training Data
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, color='blue', marker='o', s=100, alpha=0.6, label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (US$)')
plt.ylabel('Predicted Price (US$)')
plt.title('Actual vs Predicted Prices (Training Data)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Residuals (Prediction Errors)
plt.subplot(1, 2, 2)
residuals = y - y_pred
plt.scatter(y_pred, residuals, color='green', marker='o', s=100, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price (US$)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


### STEP - 8
# Summary of the model
print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
print("\nLinear Regression Equation:")
print(f"Price = {model.intercept_:.2f} + {model.coef_[0]:.2f}*Area + {model.coef_[1]:.2f}*Bedrooms + {model.coef_[2]:.2f}*Age")
print(f"\nInterpretation:")
print(f"  - For every additional sq ft: ${model.coef_[0]:.2f} increase in price")
print(f"  - For every additional bedroom: ${model.coef_[1]:.2f} change in price")
print(f"  - For every additional year of age: ${model.coef_[2]:.2f} change in price")
print(f"\n  Model explains {r2*100:.2f}% of the variance in home prices")
