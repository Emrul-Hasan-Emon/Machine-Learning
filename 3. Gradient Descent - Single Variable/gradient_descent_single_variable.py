### STEP - 1
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### STEP - 2
# Importing the dataset
df = pd.read_csv('datasets/homeprices.csv')

# Preview the data
print("Training Dataset:")
print(df.head())
print()


### STEP - 3
# Preparing the data
# Converts pandas DataFrame columns to numpy arrays
# reshapes the 1D array into a 2D column vector
X = df['area'].values.reshape(-1, 1)  # Feature (area)
y = df['price'].values.reshape(-1, 1)  # Target (price)

# Normalize the data for better gradient descent convergence
X_mean = X.mean()   # Average area
X_std = X.std()     # Standard deviation of area
y_mean = y.mean()   # Average price
y_std = y.std()     # Standard deviation of price

X_normalized = (X - X_mean) / X_std
y_normalized = (y - y_mean) / y_std


### STEP - 4
# Implementing Gradient Descent
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Implement gradient descent algorithm for linear regression.
    
    Parameters:
    X: Input features (normalized)
    y: Target values (normalized)
    learning_rate: Step size for gradient descent
    iterations: Number of iterations
    
    Returns:
    m: Slope (coefficient)
    b: Intercept
    cost_history: List of cost values over iterations
    """
    m_current = 0.0
    b_current = 0.0
    n = len(X)
    cost_history = []
    
    for i in range(iterations):
        # Predictions
        y_pred = m_current * X + b_current
        
        # Calculate cost (Mean Squared Error)
        cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)
        
        # Calculate gradients
        dm = (1 / n) * np.sum((y_pred - y) * X)
        db = (1 / n) * np.sum(y_pred - y)
        
        # Update parameters
        m_current = m_current - learning_rate * dm
        b_current = b_current - learning_rate * db
        
        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}: Cost = {cost:.6f}")
    
    return m_current, b_current, cost_history


# Train the model using gradient descent
print("Training the model using Gradient Descent...")
learning_rate = 0.01
iterations = 1000
m, b, cost_history = gradient_descent(X_normalized, y_normalized, learning_rate, iterations)

print(f"\nTraining completed!")
print(f"Final Slope (m): {m:.6f}")
print(f"Final Intercept (b): {b:.6f}")
print()


### STEP - 5
# Denormalize parameters to get actual coefficients
# y = m*x + b (in normalized form)
# y_actual = m * ((x - X_mean)/X_std) + b + y_mean * y_std
# Convert to actual: price = slope * area + intercept

slope_actual = m * (y_std / X_std)
intercept_actual = b * y_std + y_mean - (m * X_mean * y_std / X_std)

print(f"Actual Slope: {slope_actual:.6f}")
print(f"Actual Intercept: {intercept_actual:.6f}")
print()


### STEP - 6
# Plotting the Cost Function
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cost_history, color='blue', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Over Iterations')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot actual data with regression line
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='red', marker='o', label='Training Data')
plt.plot(X, slope_actual * X + intercept_actual, color='blue', linewidth=2, label='Fitted Line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (US$)')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('datasets/gradient_descent_training.png')
plt.show()


### STEP - 7
# Test with the areas dataset
areas_df = pd.read_csv('datasets/areas.csv')
print("Test Dataset:")
print(areas_df.head())
print()

# Make predictions
X_test = areas_df['area'].values.reshape(-1, 1)
predicted_prices = slope_actual * X_test + intercept_actual

# Add predictions to dataframe
areas_df['predicted_price'] = predicted_prices.flatten()

print("Predictions for Test Dataset:")
print(areas_df)
print()


### STEP - 8
# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the training data
plt.scatter(df['area'], df['price'], color='blue', label='Training Data', marker='o', s=50)

# Plot the predicted values from test set
plt.scatter(areas_df['area'], areas_df['predicted_price'], color='red', 
            label='Predicted Prices (Test)', marker='x', s=100, linewidth=2)

# Plot the regression line
plt.plot(X, slope_actual * X + intercept_actual, color='green', linestyle='--', 
         linewidth=2, alpha=0.7, label='Regression Line')

plt.xlabel('Area (sq ft)')
plt.ylabel('Price (US$)')
plt.title('Home Price Prediction using Gradient Descent')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('datasets/gradient_descent_predictions.png')
plt.show()
