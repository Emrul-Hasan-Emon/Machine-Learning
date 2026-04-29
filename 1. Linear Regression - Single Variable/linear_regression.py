### STEP - 1
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive
from sklearn.linear_model import LinearRegression


### STEP - 2
# Importing the dataset from google drive
# Mount the drive
drive.mount('/content/drive')

# Define the file path
file_path = '/content/drive/MyDrive/Machine Learning/Datasets/homeprices.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Preview the data
print(df.head())

# Plotting The Dataset
plt.figure(figsize=(10, 6))
plt.scatter(df['area'], df['price'], color='blue', marker='o', label='Home Prices')

plt.title('Area vs Price Distribution')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (US$)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save the plot
plt.savefig('home_prices_graph.png')
plt.show()


### STEP - 3
# Preparing the data
# X must be a 2D array for scikit-learn
X = df[['area']] 
y = df['price']


### STEP - 4
# Training the Linear Regression model
model = LinearRegression()
model.fit(X, y)


### STEP - 5
# Predicting the price of a home with an area of 2500 sq ft
area_to_predict = 2500
prediction = model.predict(pd.DataFrame({'area': [area_to_predict]}))
print(f"Predicted price for area {area_to_predict}: {prediction[0]:.2f}")

# Plotting the Results
plt.figure(figsize=(10, 6))

# Plot actual data points
plt.scatter(df.area, df.price, color='red', marker='+', label='Actual Data')

# Plot the regression line
plt.plot(df.area, model.predict(df[['area']]), color='blue', label='Linear Regression Line')

# Labels and Title
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (US$)')
plt.title('Home Price Prediction using Linear Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()


# Test By Dataset
# Load the new areas dataset for testing
new_areas_path = '/content/drive/MyDrive/Machine Learning/Datasets/areas.csv'
areas_df = pd.read_csv(new_areas_path)
print(areas_df)


# Generate predictions
# We use the existing 'model' trained in previous steps
areas_df['predicted_price'] = model.predict(areas_df[['area']])

# Display the results
print(areas_df)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the original training data (the 'actual' values)
plt.scatter(df['area'], df['price'], color='blue', label='Training Data')

# Plot the new predicted data (the 'results')
plt.scatter(areas_df['area'], areas_df['predicted_price'], color='red', marker='x', s=100, label='Predicted Prices')

# Plot the regression line for visual context
plt.plot(df['area'], model.predict(df[['area']]), color='green', linestyle='--', alpha=0.5, label='Regression Line')

# Chart labels and formatting
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Actual vs Predicted Home Prices')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()