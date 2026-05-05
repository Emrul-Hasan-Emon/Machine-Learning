### STEP - 1
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### STEP - 2
# Importing the dataset
df = pd.read_csv('datasets/homeprices_multiple_variables.csv')

# Preview the data
print("Training Dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Statistics:")
print(df.describe())
print()


### STEP - 3
# Preparing the data
# X contains all independent variables (features)
X = df[['area', 'bedrooms', 'age']].values  # Shape: (n_samples, n_features)
y = df['price'].values.reshape(-1, 1)        # Shape: (n_samples, 1)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of features: {X.shape[1]}")
print()


### STEP - 4
# Normalize the data for better gradient descent convergence
# Store means and stds for denormalization later
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
y_mean = y.mean()
y_std = y.std()

X_normalized = (X - X_mean) / X_std
y_normalized = (y - y_mean) / y_std

print("Normalization Parameters:")
print(f"X Mean: {X_mean}")
print(f"X Std: {X_std}")
print(f"Y Mean: {y_mean}")
print(f"Y Std: {y_std}")
print()


### STEP - 5
# Implementing Gradient Descent for Multiple Variables
def gradient_descent_multiple(X, y, learning_rate=0.01, iterations=1000):
    """
    Implement gradient descent algorithm for multiple linear regression.
    
    Parameters:
    X: Input features (normalized) - Shape: (n_samples, n_features)
    y: Target values (normalized) - Shape: (n_samples, 1)
    learning_rate: Step size for gradient descent
    iterations: Number of iterations
    
    Returns:
    weights: Coefficients for each feature
    bias: Intercept term
    cost_history: List of cost values over iterations
    """
    n_samples, n_features = X.shape
    
    # Initialize weights and bias
    weights = np.zeros((n_features, 1))
    bias = 0.0
    cost_history = []
    
    for i in range(iterations):
        # 1. Make predictions: y_pred = X @ weights + bias
        y_pred = X.dot(weights) + bias
        
        # 2. Calculate cost (Mean Squared Error)
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)
        
        # 3. Calculate gradients
        # Gradient for weights: dJ/dw = (1/n) * X.T @ (y_pred - y)
        dw = (1 / n_samples) * X.T.dot(y_pred - y)
        
        # Gradient for bias: dJ/db = (1/n) * sum(y_pred - y)
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        # 4. Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}: Cost = {cost:.6f}")
    
    return weights, bias, cost_history


# Train the model using gradient descent
print("Training the model using Gradient Descent...")
learning_rate = 0.01
iterations = 1000
weights, bias, cost_history = gradient_descent_multiple(
    X_normalized, 
    y_normalized, 
    learning_rate, 
    iterations
)

print(f"\nTraining completed!")
print(f"\nNormalized Model Parameters:")
print(f"Weights (normalized): {weights.flatten()}")
print(f"Bias (normalized): {bias:.6f}")
print()


### STEP - 6
# Denormalize parameters to get actual coefficients
# y_normalized = weights.T @ X_normalized + bias
# y = y_std * y_normalized + y_mean
# y = y_std * (weights.T @ X_normalized + bias) + y_mean
# y = y_std * (weights.T @ ((X - X_mean) / X_std) + bias) + y_mean
# y = (y_std / X_std) * weights.T @ (X - X_mean) + y_std * bias + y_mean
# y = (y_std / X_std) * weights.T @ X - (y_std / X_std) * weights.T @ X_mean + y_std * bias + y_mean

weights_actual = (y_std / X_std).reshape(-1, 1) * weights
bias_actual = y_std * bias + y_mean - np.sum((y_std / X_std) * weights.flatten() * X_mean)

print("Actual Model Parameters:")
print(f"Coefficients:")
print(f"  Area coefficient: {weights_actual[0, 0]:.2f}")
print(f"  Bedrooms coefficient: {weights_actual[1, 0]:.2f}")
print(f"  Age coefficient: {weights_actual[2, 0]:.2f}")
print(f"Intercept: {bias_actual:.2f}")
print()


### STEP - 7
# Plotting the Cost Function
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cost_history, color='blue', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Over Iterations')
plt.grid(True, linestyle='--', alpha=0.7)

# Predictions on training data
plt.subplot(1, 2, 2)
y_pred_normalized = X_normalized.dot(weights) + bias
y_pred_actual = y_std * y_pred_normalized + y_mean

plt.scatter(y, y_pred_actual, color='blue', marker='o', s=100, alpha=0.6, label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (US$)')
plt.ylabel('Predicted Price (US$)')
plt.title('Actual vs Predicted Prices (Training Data)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('gradient_descent_multiple_training.png')
plt.show()


### STEP - 8
# Make predictions on training data and calculate metrics
y_pred_normalized = X_normalized.dot(weights) + bias
y_pred_actual = y_std * y_pred_normalized + y_mean

# Calculate R² score
ss_tot = np.sum((y - y.mean()) ** 2)
ss_res = np.sum((y - y_pred_actual) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Calculate RMSE
rmse = np.sqrt(np.mean((y - y_pred_actual) ** 2))

print("="*60)
print("MODEL PERFORMANCE METRICS (Training Data)")
print("="*60)
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print()


### STEP - 9
# Test with the testing dataset
test_df = pd.read_csv('datasets/testing.csv')
print("Test Dataset:")
print(test_df.head())
print()

# Extract features from test data
X_test = test_df[['area', 'bedrooms', 'age']].values

# Normalize test data using training set statistics
X_test_normalized = (X_test - X_mean) / X_std

# Make predictions on test data
y_test_pred_normalized = X_test_normalized.dot(weights) + bias
y_test_pred_actual = y_std * y_test_pred_normalized + y_mean

# Add predictions to test dataframe
test_df['predicted_price'] = y_test_pred_actual.flatten()

print("Predictions for Test Dataset:")
print(test_df)
print()


### STEP - 10
# Visualizing results
plt.figure(figsize=(14, 5))

# Plot 1: Cost function convergence
plt.subplot(1, 2, 1)
plt.plot(cost_history, color='blue', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent: Cost Function Convergence')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Training vs Predictions
plt.subplot(1, 2, 2)
sample_indices = np.arange(len(y))
plt.bar(sample_indices - 0.2, y.flatten(), width=0.4, label='Actual Prices', alpha=0.7, color='blue')
plt.bar(sample_indices + 0.2, y_pred_actual.flatten(), width=0.4, label='Predicted Prices', alpha=0.7, color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Price (US$)')
plt.title('Actual vs Predicted Prices (Training Data)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.savefig('gradient_descent_multiple_results.png')
plt.show()


### STEP - 11
# Print model summary
print("="*60)
print("GRADIENT DESCENT MODEL SUMMARY")
print("="*60)
print(f"\nLinear Regression Equation:")
print(f"Price = {bias_actual:.2f} + {weights_actual[0, 0]:.2f}×Area + {weights_actual[1, 0]:.2f}×Bedrooms + {weights_actual[2, 0]:.2f}×Age")
print(f"\nInterpretation:")
print(f"  • For every additional sq ft: ${weights_actual[0, 0]:.2f} increase in price")
print(f"  • For every additional bedroom: ${weights_actual[1, 0]:.2f} change in price")
print(f"  • For every additional year of age: ${weights_actual[2, 0]:.2f} change in price")
print(f"  • Model explains {r2*100:.2f}% of the variance in home prices")
print()

print("="*60)
print("SAMPLE PREDICTIONS")
print("="*60)
for idx, row in test_df.iterrows():
    if idx < 5:
        print(f"House {idx + 1}: Area={row['area']}, Bedrooms={row['bedrooms']}, Age={row['age']}")
        print(f"  → Predicted Price: ${row['predicted_price']:,.2f}")
        print()
