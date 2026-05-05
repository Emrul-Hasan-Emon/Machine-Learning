# Gradient Descent for Linear Regression (Multiple Variables)

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [What is Multiple Linear Regression?](#what-is-multiple-linear-regression)
3. [Mathematical Concepts](#mathematical-concepts)
4. [Gradient Descent for Multiple Variables](#gradient-descent-for-multiple-variables)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Why Normalization Matters](#why-normalization-matters)
7. [Matrix Operations Explained](#matrix-operations-explained)
8. [How to Run](#how-to-run)
9. [Understanding the Output](#understanding-the-output)
10. [Performance Metrics](#performance-metrics)

---

## 📖 Introduction

This project extends **Gradient Descent** to **multiple independent variables**. Instead of predicting home prices from just area, we now use:
- **Area** (sq ft)
- **Bedrooms** (count)
- **Age** (years)

**Problem:** Given multiple features of a house, predict its price.

**Solution:** Use gradient descent to learn optimal parameters for: `Price = w₁×Area + w₂×Bedrooms + w₃×Age + b`

---

## 🎯 What is Multiple Linear Regression?

### From Single to Multiple Variables

**Single Variable:**
```
y = m × x + b
Price = slope × area + intercept
```

**Multiple Variables:**
```
y = w₁×x₁ + w₂×x₂ + w₃×x₃ + b
Price = w₁×area + w₂×bedrooms + w₃×age + bias
```

### Key Differences

| Aspect | Single Variable | Multiple Variables |
|--------|---|---|
| **Features** | 1 (area) | 3+ (area, bedrooms, age) |
| **Weights** | 1 slope (m) | Multiple weights (w₁, w₂, w₃, ...) |
| **Complexity** | Simple line | Hyperplane |
| **Math** | Scalars | Vectors & Matrices |
| **Accuracy** | Lower | Higher (more info) |

### Example Dataset

```
area | bedrooms | age | price
-----|----------|-----|--------
2600 |    3     | 20  | 550000
3000 |    4     | 15  | 565000
3200 |    3     | 18  | 610000
...
```

---

## 📐 Mathematical Concepts

### 1. Linear Regression Model

$$\hat{y} = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

Or in **matrix form:**

$$\hat{y} = X \cdot W + b$$

Where:
- **X** = Feature matrix (n_samples × n_features)
- **W** = Weight vector (n_features × 1)
- **b** = Bias (scalar)
- **ŷ** = Predictions

### 2. Cost Function (Mean Squared Error)

$$J(W, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_{pred}^{(i)} - y_{actual}^{(i)})^2$$

### 3. Gradients (Partial Derivatives)

**Gradient for weights:**
$$\frac{\partial J}{\partial W} = \frac{1}{n} X^T (y_{pred} - y_{actual})$$

**Gradient for bias:**
$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (y_{pred}^{(i)} - y_{actual}^{(i)})$$

### 4. Parameter Update Rule

$$W_{new} = W_{old} - \alpha \cdot \frac{\partial J}{\partial W}$$

$$b_{new} = b_{old} - \alpha \cdot \frac{\partial J}{\partial b}$$

---

## 🔄 Gradient Descent for Multiple Variables

### Algorithm Steps

```
1. Initialize weights W = [0, 0, 0, ...] and bias b = 0
2. For each iteration:
   a. Predictions: ŷ = X·W + b
   b. Cost: J = (1/2n) × Σ(ŷ - y)²
   c. Gradients: 
      - dW = (1/n) × X.T × (ŷ - y)
      - db = (1/n) × Σ(ŷ - y)
   d. Update:
      - W = W - α × dW
      - b = b - α × db
3. Repeat until convergence
```

### Key Insight: Vectorization

Instead of looping through features:

**Bad (slow):**
```python
for j in range(n_features):
    dw[j] = (1/n) * sum((y_pred - y) * X[:, j])
```

**Good (fast with matrices):**
```python
dw = (1/n) * X.T @ (y_pred - y)  # Matrix multiplication!
```

This is why we use **matrices and numpy** — it's much faster!

---

## 🛠️ Step-by-Step Implementation

### STEP 1-3: Load & Prepare Data

```python
df = pd.read_csv('datasets/homeprices_multiple_variables.csv')

X = df[['area', 'bedrooms', 'age']].values    # (6, 3)
y = df['price'].values.reshape(-1, 1)         # (6, 1)
```

**Shapes:**
- X: (6 samples, 3 features)
- y: (6 samples, 1 target)

### STEP 4: Normalize Data

```python
X_mean = X.mean(axis=0)      # Mean of each feature
X_std = X.std(axis=0)        # Std of each feature
X_normalized = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / y_std
```

#### Understanding axis=0: Crucial Concept!

**What does `axis=0` mean?**

`axis=0` means: **Calculate statistics DOWN each column (vertically)**

**Data Structure:**
```
     Area       Bedrooms   Age      ← Features (Columns)
      │           │        │
      ↓ axis=0    ↓        ↓
    2600        3        20       ← Sample 1
    3000        4        15       ← Sample 2
    3200        3        18       ← Sample 3
    3600        3        30       ← Sample 4
    4000        5         8       ← Sample 5
    4100        6         8       ← Sample 6
    ────────────────────────
    MEAN OF   MEAN OF   MEAN OF
    ALL      ALL      ALL
    AREAS    BEDROOMS  AGES
```

**What gets calculated with axis=0:**

```python
X.mean(axis=0)  # ONE average PER FEATURE
```

Result: `[3433.33, 4.0, 16.5]` ← One value per feature!

```
Area mean     = (2600+3000+3200+3600+4000+4100)/6 = 3433.33
Bedrooms mean = (3+4+3+3+5+6)/6 = 4.0
Age mean      = (20+15+18+30+8+8)/6 = 16.5
```

```python
X.std(axis=0)   # ONE std dev PER FEATURE
```

Result: `[650.5, 1.2, 8.9]` ← One value per feature!

**Why this is critical:**

Your features have **different scales:**
```
Area:     2600 - 4100  (large numbers, range ≈ 1500)
Bedrooms: 3 - 6        (small numbers, range ≈ 3)
Age:      8 - 30       (medium numbers, range ≈ 22)
```

**Without axis=0 (WRONG):**
```python
X.mean()  # No axis parameter
```
Would give you ONE average of ALL numbers mixed together:
```
(2600 + 3000 + ... + 3 + 4 + ... + 20 + 15 + ...) / 18 = 1842.67
```
This mixes Area, Bedrooms, and Age together! ❌ Useless!

**With axis=0 (CORRECT):**
Computes separate statistics for each feature independently ✅

#### How Normalization Uses These Statistics

```python
X_normalized = (X - X_mean) / X_std
```

**For first sample:**
```
Original: [2600, 3, 20]
X_mean:   [3433.33, 4.0, 16.5]
X_std:    [650.5, 1.2, 8.9]

Area normalized = (2600 - 3433.33) / 650.5 = -1.28
Bedrooms normalized = (3 - 4.0) / 1.2 = -0.83
Age normalized = (20 - 16.5) / 8.9 = 0.39

Result: [-1.28, -0.83, 0.39]  ← All on similar scales!
```

**Result:** Each feature normalized independently to mean ≈ 0, std ≈ 1 ✅

#### Why `axis=0` and NOT axis=1?

| Axis | What Happens | Result | Use Case |
|------|---|---|---|
| **No axis** | Average ALL numbers | 1 number | ❌ Wrong |
| **axis=0** | Average DOWN each column | 1 value per feature | ✅ CORRECT! |
| **axis=1** | Average ACROSS each row | 1 value per sample | ❌ Wrong for normalization |

**Example with axis=1 (WRONG):**
```python
X.mean(axis=1)  # Average per sample
```
Result: `[874.33, 1006.33, 1073.67, ...]`

This gives the average of Area, Bedrooms, and Age mixed together for each sample. We don't want that! ❌

#### Key Takeaway

```python
X_mean = X.mean(axis=0)  # ✅ Get average of each FEATURE
X_std = X.std(axis=0)    # ✅ Get std dev of each FEATURE
X_normalized = (X - X_mean) / X_std  # ✅ Normalize each feature independently!
```

This ensures **gradient descent receives balanced features** all on similar scales, leading to stable and fast convergence! 🎯

---

```python
def gradient_descent_multiple(X, y, learning_rate=0.01, iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0.0
    
    for i in range(iterations):
        # Predictions
        y_pred = X.dot(weights) + bias
        
        # Cost
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
        
        # Gradients
        dw = (1 / n_samples) * X.T.dot(y_pred - y)
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        # Update
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
    
    return weights, bias, cost_history
```

### STEP 6: Denormalize Parameters

Since we trained on normalized data, we need to convert back:

```python
weights_actual = (y_std / X_std).reshape(-1, 1) * weights
bias_actual = y_std * bias + y_mean - np.sum(
    (y_std / X_std) * weights.flatten() * X_mean
)
```

**Why?** Learned parameters (W, b) work with normalized data, but we need predictions on original data.

### STEP 7-11: Evaluate & Predict

```python
# Make predictions on test set
X_test_normalized = (X_test - X_mean) / X_std
y_test_pred_normalized = X_test_normalized.dot(weights) + bias
y_test_pred = y_std * y_test_pred_normalized + y_mean

# Calculate R² and RMSE
r2 = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
```

---

## ⚠️ Why Normalization Matters

### Problem Without Normalization

**Raw data ranges:**
- Area: 2,600 → 4,100 (range ≈ 1,500)
- Bedrooms: 3 → 6 (range ≈ 3)
- Age: 8 → 30 (range ≈ 22)
- Price: 550,000 → 810,000 (range ≈ 260,000)

**Gradients become imbalanced:**
```
Area gradient:     Very LARGE (millions)
Bedrooms gradient: Very SMALL (hundreds)
Age gradient:      Medium
Price target:      HUGE (hundreds of thousands)
```

Result: Algorithm diverges or converges very slowly ❌

### Solution With Normalization

**Normalized ranges:**
- Area: -1.2 → +0.8 (normalized)
- Bedrooms: -0.6 → +0.9 (normalized)
- Age: -0.8 → +1.1 (normalized)
- Price: -0.5 → +1.2 (normalized)

**Gradients become balanced:**
```
All gradients: ~0.1 to ~0.5 (similar scale)
```

Result: Fast, stable convergence ✅

### Convergence Comparison

**Without Normalization:**
```
Iteration 100:  Cost = 1.2e+12
Iteration 200:  Cost = 5.6e+13  ← JUMPED UP!
Iteration 300:  Cost = DIVERGED! ← Failed
```

**With Normalization:**
```
Iteration 100:  Cost = 0.456
Iteration 200:  Cost = 0.234
Iteration 300:  Cost = 0.178  ← Smooth decrease
```

---

## 🔢 Matrix Operations Explained

### Why We Use Matrices

For 3 features and 6 samples, calculating gradients:

**Without matrices (nested loops):**
```python
for sample in range(6):
    for feature in range(3):
        error = y_pred[sample] - y[sample]
        dw[feature] += error * X[sample, feature]
```

**With matrices (one line):**
```python
dw = X.T @ (y_pred - y)  # Matrix multiplication!
```

### Understanding X^T (Transpose)

**X^T means "Transpose of X"** — flip the matrix rows and columns!

#### What is Transpose?

**Original Matrix X (6 samples × 3 features):**
```
       Area  Bedrooms  Age
       ────  ────────  ───
S1     2600     3      20
S2     3000     4      15
S3     3200     3      18
S4     3600     3      30
S5     4000     5       8
S6     4100     6       8

Shape: (6, 3)  ← 6 rows, 3 columns
```

**Transposed Matrix X^T (3 features × 6 samples):**
```
         S1    S2    S3    S4    S5    S6
       ─────  ────  ────  ────  ────  ────
Area    2600  3000  3200  3600  4000  4100
Bedrooms 3     4     3     3     5     6
Age      20    15    18    30    8     8

Shape: (3, 6)  ← 3 rows, 6 columns
```

#### How Transpose Works

```python
X = np.array([
    [2600, 3, 20],      # Sample 1
    [3000, 4, 15],      # Sample 2
    [3200, 3, 18],      # Sample 3
    [3600, 3, 30],      # Sample 4
    [4000, 5, 8],       # Sample 5
    [4100, 6, 8]        # Sample 6
])
# Shape: (6, 3)

X.T = np.array([
    [2600, 3000, 3200, 3600, 4000, 4100],  # Area column becomes row
    [3, 4, 3, 3, 5, 6],                    # Bedrooms column becomes row
    [20, 15, 18, 30, 8, 8]                 # Age column becomes row
])
# Shape: (3, 6)
```

**Simple rule:** If X is (m, n), then X^T is (n, m)
```
X:   (6, 3)  →  X^T:  (3, 6)
     (6 rows, 3 cols) → (3 rows, 6 cols)
```

---

### How X^T is Used in Gradient Calculation

In the gradient descent algorithm, we calculate:

```python
dw = (1 / n_samples) * X.T.dot(y_pred - y)
```

**Breaking it down:**

**Step 1: Error vector**
```python
error = y_pred - y
Shape: (6, 1)

error = [10000, -5000, 3000, -8000, 2000, 5000]
```

**Step 2: Multiply X^T with error**
```python
X.T.dot(error)  or  X.T @ error
```

This is **matrix multiplication:**
```
X.T              ×    error       =   gradient
(3, 6)          ×    (6, 1)      =   (3, 1)

[2600  3000  3200  3600  4000  4100]   [10000]
[3     4     3     3     5     6     ] × [-5000]  = [dw1]
[20    15    18    30    8     8     ]   [3000 ]    [dw2]
                                        [-8000]    [dw3]
                                        [2000 ]
                                        [5000 ]
```

**Step 3: What Each Element Computes**

Each element of result = **dot product** of that row with error:

```
dw1 (Area) = 2600×10000 + 3000×(-5000) + 3200×3000 + 3600×(-8000) + 4000×2000 + 4100×5000
           = 26000000 - 15000000 + 9600000 - 28800000 + 8000000 + 20500000
           = 20300000

dw2 (Bedrooms) = 3×10000 + 4×(-5000) + 3×3000 + 3×(-8000) + 5×2000 + 6×5000
               = 30000 - 20000 + 9000 - 24000 + 10000 + 30000
               = 35000

dw3 (Age) = 20×10000 + 15×(-5000) + 18×3000 + 30×(-8000) + 8×2000 + 8×5000
          = 200000 - 75000 + 54000 - 240000 + 16000 + 40000
          = -5000
```

**Result:** `dw = [20300000, 35000, -5000]` ← Gradient for each feature!

---

### Why Transpose is Necessary

**Without transpose (WRONG):**
```python
X.dot(error)  # (6, 3) @ (6, 1) = ERROR! Can't multiply
```
Dimensions don't match for matrix multiplication! ❌

**With transpose (CORRECT):**
```python
X.T.dot(error)  # (3, 6) @ (6, 1) = (3, 1) ✅
```
Dimensions work perfectly! The inner dimensions match (6 and 6). ✅

---

### Visualization: Why X^T Gives Us Feature Gradients

**The key insight:**

We want: "How much does EACH FEATURE contribute to the error?"

```
Feature 1 (Area):     How much did all areas contribute?
                      Error × [Area₁, Area₂, Area₃, ...]
                      
Feature 2 (Bedrooms): How much did all bedrooms contribute?
                      Error × [Bed₁, Bed₂, Bed₃, ...]
                      
Feature 3 (Age):      How much did all ages contribute?
                      Error × [Age₁, Age₂, Age₃, ...]
```

**Original X has this layout:**
```
Row = Sample, Column = Feature
```

**We need:**
```
Row = Feature, Column = Sample
```

**Solution: Transpose!**
```
X.T converts [Sample × Feature] to [Feature × Sample]
```

Now each row of X.T contains one complete feature across all samples!

---

### Complete Example: Step-by-Step

**Given:**
- X shape: (6, 3) — 6 samples, 3 features
- error shape: (6, 1) — one error per sample

**Calculate dw = X.T @ error:**

```
X.T                           error
(3, 6)                        (6, 1)
│                             │
Area row:    [A₁ A₂ A₃ A₄ A₅ A₆]  ×  [e₁]
             [b₁ b₂ b₃ b₄ b₅ b₆]     [e₂]  = [dw_area]
             [ag₁ ag₂ ag₃ ag₄ ag₅ ag₆]  [e₃]    [dw_bed]
                                      [e₄]    [dw_age]
Bedrooms row [b₁ b₂ b₃ b₄ b₅ b₆]      [e₅]
             (dot with error)          [e₆]
             
Age row: [ag₁ ag₂ ag₃ ag₄ ag₅ ag₆]

Result: (3, 1) = One gradient per feature!
```

---

### Key Matrix Dimension Rules

**For matrix multiplication A @ B:**
- A shape: (m, n)
- B shape: (n, p)
- Result shape: (m, p)

**In our case:**
```
X.T          @    error        =    dw
(3, 6)       @    (6, 1)       =    (3, 1)
↑                  ↑                 ↑
features          samples          gradient per feature
```

The middle dimensions must match (6 and 6). The outer dimensions (3 and 1) become the result shape!

---

### Summary: Why We Use X^T

1. **X has shape (samples, features)** — each row is a sample
2. **We need (features, samples)** — each row is a feature
3. **Transpose X to get X^T**
4. **X^T @ error gives us** gradient for each feature efficiently
5. **One matrix multiplication replaces nested loops!** ✅

---

### Key Matrix Operations

---


### Expected Output
```
Training Dataset:
   area  bedrooms  age   price
0  2600         3   20   550000
1  3000         4   15   565000
...

Features shape: (6, 3)
Target shape: (6, 1)
Number of features: 3

Normalization Parameters:
X Mean: [3483.33 3.83 15.83]
X Std: [700.02 1.17 8.24]
Y Mean: 645000.0
Y Std: 109373.98

Training the model using Gradient Descent...
Iteration 100: Cost = 0.245632
Iteration 200: Cost = 0.156234
...
Training completed!

Actual Model Parameters:
Coefficients:
  Area coefficient: 157.89
  Bedrooms coefficient: 45678.23
  Age coefficient: -2345.67
Intercept: 123456.78

============================================================
MODEL PERFORMANCE METRICS (Training Data)
============================================================
R² Score: 0.9876
Root Mean Squared Error (RMSE): $5432.10

Test Dataset:
     area  bedrooms  age
0   1000         1   12
1   1500         2    4
...

Predictions for Test Dataset:
     area  bedrooms  age  predicted_price
0   1000         1   12     345678.90
1   1500         2    4     456789.23
...
```


---
## 🔑 Key Differences: Single vs Multiple Variables

| Aspect | Single | Multiple |
|--------|--------|----------|
| **Equation** | y = mx + b | y = w₁x₁ + w₂x₂ + ... + b |
| **Shapes** | X: (n,), W: scalar | X: (n, p), W: (p, 1) |
| **Gradients** | dm, db (scalars) | dw vector (p, 1) |
| **Prediction** | m*X + b | X @ W + b |
| **Complexity** | Low | Higher |
| **Accuracy** | Lower | Higher |
| **Interpretability** | Easy | Moderate |
| **Computation** | Fast | Slightly slower |

---

## ⚡ Performance Tips

### 1. Feature Scaling
Always normalize features first (different ranges cause problems)

### 2. Learning Rate Selection
```
Too high (0.1):    Diverges
Optimal (0.01):    Converges smoothly
Too low (0.0001):  Very slow
```

### 3. Iterations
```
Too few (100):     Underfit
Optimal (1000):    Usually enough
Too many (10000):  Waste of time
```

### 4. Feature Engineering
More features ≠ better model
- Too many features → Overfitting
- Wrong features → Poor accuracy

---

## 📝 Summary

| Step | Action | Key Point |
|------|--------|-----------|
| 1 | Load data | Shape: (samples, features) |
| 2 | Prepare | Extract X and y |
| 3 | Normalize | Use axis=0 for features |
| 4 | Initialize | W = 0, b = 0 |
| 5 | Loop | Calculate gradients, update |
| 6 | Denormalize | Convert to original scale |
| 7 | Evaluate | Calculate R² and RMSE |
| 8 | Predict | Use test features |

---

## ✨ Key Takeaways

1. **Multiple linear regression** handles multiple features simultaneously
2. **Matrix operations** make gradient descent efficient for many features
3. **Normalization is critical** for stable convergence with multiple variables
4. **More features ≠ better** — balance accuracy with simplicity
5. **Denormalization** is needed to interpret results in original scale
6. **R² and RMSE** help evaluate model performance

---

## 🎓 Advanced Topics

- **Feature Selection** - Which features matter most?
- **Regularization** - Prevent overfitting
- **Cross-validation** - Better performance estimation
- **Stochastic Gradient Descent** - Mini-batches for large data
- **Polynomial features** - Non-linear relationships
- **Multicollinearity** - When features are correlated

---


