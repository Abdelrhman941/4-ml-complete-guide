# 📈 Linear Regression

Linear Regression is the foundation of supervised learning and statistical modeling. It models the relationship between a dependent variable and independent variables by fitting a linear equation to observed data.

## 🎯 Core Concept

Linear regression assumes a linear relationship between input features (X) and the target variable (y):

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- `y` = target variable (dependent variable)
- `x₁, x₂, ..., xₙ` = input features (independent variables)
- `β₀` = intercept (bias term)
- `β₁, β₂, ..., βₙ` = coefficients (weights)
- `ε` = error term (noise)

## 🧮 Mathematical Foundation

### Simple Linear Regression (One Feature)
```
y = β₀ + β₁x + ε
```

**Goal:** Find the best line that minimizes the sum of squared errors.

### Multiple Linear Regression (Multiple Features)
```
y = Xβ + ε
```

Where:
- `X` = feature matrix (n × p)
- `β` = coefficient vector (p × 1)
- `y` = target vector (n × 1)

### Cost Function (Mean Squared Error)
```
J(β) = (1/2m) Σ(hβ(xᵢ) - yᵢ)²
```

### Normal Equation (Analytical Solution)
```
β = (XᵀX)⁻¹Xᵀy
```

## 🔧 Key Assumptions

1. **Linearity**: The relationship between X and y is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

## ✅ Advantages

- **Simplicity**: Easy to understand and implement
- **Interpretability**: Coefficients show feature importance and direction
- **Fast Training**: Closed-form solution available
- **No Hyperparameters**: No tuning required for basic version
- **Baseline Model**: Good starting point for regression problems
- **Statistical Inference**: Confidence intervals and p-values available

## ❌ Disadvantages

- **Linear Assumption**: Can't capture non-linear relationships
- **Sensitive to Outliers**: Outliers can significantly affect the model
- **Feature Scaling**: Performance affected by feature scales
- **Overfitting**: With many features relative to samples
- **Multicollinearity**: Unstable when features are correlated

## 🎯 When to Use Linear Regression

### ✅ Good For:
- **Linear Relationships**: When the relationship is approximately linear
- **Interpretability**: When you need to explain the model
- **Baseline Model**: Quick initial model to establish performance
- **Small Datasets**: Works well with limited data
- **Feature Importance**: Understanding which features matter

### ❌ Avoid When:
- **Non-linear Patterns**: Complex relationships between variables
- **Many Outliers**: Data contains significant outliers
- **High Dimensionality**: More features than samples
- **Non-continuous Target**: Target variable is categorical

## 📊 Types of Linear Regression

### 1. Simple Linear Regression
- **Use Case**: One independent variable
- **Equation**: y = β₀ + β₁x
- **Example**: Price vs. Size of house

### 2. Multiple Linear Regression
- **Use Case**: Multiple independent variables
- **Equation**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- **Example**: House price vs. size, location, age

### 3. Polynomial Regression
- **Use Case**: Non-linear relationships
- **Equation**: y = β₀ + β₁x + β₂x² + ... + βₙxⁿ
- **Note**: Still linear in parameters

## 🔧 Regularized Versions

### Ridge Regression (L2 Regularization)
```
J(β) = MSE + α Σβᵢ²
```
- **Purpose**: Prevents overfitting by penalizing large coefficients
- **Effect**: Shrinks coefficients toward zero
- **When to use**: Many correlated features

### Lasso Regression (L1 Regularization)
```
J(β) = MSE + α Σ|βᵢ|
```
- **Purpose**: Feature selection + regularization
- **Effect**: Can set coefficients exactly to zero
- **When to use**: Feature selection is important

### Elastic Net (L1 + L2)
```
J(β) = MSE + α₁ Σ|βᵢ| + α₂ Σβᵢ²
```
- **Purpose**: Combines benefits of Ridge and Lasso
- **When to use**: Many features, some groups are correlated

## 📈 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | (1/n)Σ\|y - ŷ\| | Average absolute error |
| **MSE** | (1/n)Σ(y - ŷ)² | Average squared error |
| **RMSE** | √MSE | Root mean squared error |
| **R²** | 1 - (SS_res/SS_tot) | Proportion of variance explained |

## 🎯 Implementation Steps

### 1. Data Preparation
```python
# Load and explore data
# Handle missing values
# Feature scaling (if needed)
# Train-test split
```

### 2. Model Training
```python
# Fit the model
# Calculate coefficients
# Make predictions
```

### 3. Model Evaluation
```python
# Calculate metrics
# Residual analysis
# Check assumptions
```

### 4. Model Interpretation
```python
# Analyze coefficients
# Feature importance
# Confidence intervals
```

## 📊 Diagnostic Plots

### 1. Residuals vs. Fitted
- **Purpose**: Check for non-linearity and heteroscedasticity
- **Good**: Random scatter around zero
- **Bad**: Patterns or funnel shapes

### 2. Q-Q Plot
- **Purpose**: Check normality of residuals
- **Good**: Points follow diagonal line
- **Bad**: Curved pattern

### 3. Residuals vs. Leverage
- **Purpose**: Identify influential outliers
- **Watch for**: Points with high leverage and high residuals

## 🎯 What You'll Learn

### Theoretical Understanding
- [ ] Mathematical derivation of normal equation
- [ ] Assumptions and their implications
- [ ] Relationship to maximum likelihood estimation
- [ ] Geometric interpretation of least squares

### Practical Implementation
- [ ] From-scratch implementation using NumPy
- [ ] Scikit-learn implementation
- [ ] Feature engineering techniques
- [ ] Handling categorical variables

### Model Evaluation
- [ ] Residual analysis
- [ ] Assumption checking
- [ ] Cross-validation for regression
- [ ] Interpreting coefficients

### Advanced Topics
- [ ] Regularized regression (Ridge, Lasso, Elastic Net)
- [ ] Polynomial features
- [ ] Interaction terms
- [ ] Dealing with multicollinearity

## 🚀 Real-World Applications

- **Economics**: Demand forecasting, price modeling
- **Marketing**: Sales prediction, campaign effectiveness
- **Finance**: Risk assessment, portfolio optimization
- **Healthcare**: Drug dosage effects, treatment outcomes
- **Real Estate**: Property valuation
- **Manufacturing**: Quality control, process optimization

---

*Linear regression: Simple, interpretable, and surprisingly powerful! 📈*
