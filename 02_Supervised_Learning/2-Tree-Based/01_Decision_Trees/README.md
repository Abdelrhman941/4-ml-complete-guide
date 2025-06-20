# 🌳 Decision Trees

Decision Trees are intuitive, interpretable machine learning models that make decisions by asking a series of questions about the data features. They work for both classification and regression problems.

## 🎯 Core Concept

A decision tree learns a series of if-else conditions on the features to predict the target variable. It creates a tree-like structure where:
- **Internal nodes** represent features/attributes
- **Branches** represent decision rules
- **Leaf nodes** represent outcomes/predictions

## 🌲 How Decision Trees Work

### Tree Structure Example
```
        Feature A < 5?
           /        \
         Yes          No
         /              \
   Feature B < 3?    Feature C < 7?
      /      \          /        \
    Yes      No       Yes        No
   /          \       /            \
Class 1    Class 2  Class 1     Class 2
```

### Decision Process
1. Start at the root node
2. Follow the decision rule (feature threshold)
3. Move to the appropriate child node
4. Repeat until reaching a leaf node
5. Return the prediction from the leaf

## 🧮 Mathematical Foundation

### Splitting Criteria

#### 1. **Gini Impurity** (Classification)
```
Gini(S) = 1 - Σ(pᵢ)²
```
Where `pᵢ` is the proportion of samples belonging to class `i`

- **Range**: 0 (pure) to 0.5 (50-50 split for binary)
- **Interpretation**: Probability of incorrectly classifying a random sample

#### 2. **Entropy** (Classification)
```
Entropy(S) = -Σ pᵢ log₂(pᵢ)
```
- **Range**: 0 (pure) to 1 (maximum uncertainty for binary)
- **Interpretation**: Measure of disorder/uncertainty

#### 3. **Mean Squared Error** (Regression)
```
MSE(S) = (1/n) Σ(yᵢ - ȳ)²
```
Where `ȳ` is the mean of target values in the set

### Information Gain
```
Information Gain = Entropy(Parent) - Weighted Average of Entropy(Children)
```

The algorithm chooses splits that maximize information gain (minimize impurity).

## 🔧 Tree Building Algorithm (CART)

### Recursive Binary Splitting
1. **For each feature and threshold**:
   - Split data into two groups
   - Calculate impurity reduction
2. **Choose the best split**:
   - Maximum information gain
   - Minimum impurity
3. **Recursively apply** to child nodes
4. **Stop when**:
   - Maximum depth reached
   - Minimum samples per leaf reached
   - No significant improvement

### Stopping Criteria
- **max_depth**: Maximum tree depth
- **min_samples_split**: Minimum samples to split a node
- **min_samples_leaf**: Minimum samples in a leaf
- **min_impurity_decrease**: Minimum impurity reduction for split

## ✅ Advantages

- **Interpretability**: Easy to understand and visualize
- **No Preprocessing**: Handles categorical and numerical features
- **Feature Selection**: Automatically selects relevant features
- **Non-linear Relationships**: Captures complex patterns
- **Missing Values**: Can handle missing data (some implementations)
- **Fast Prediction**: Quick inference once trained
- **No Assumptions**: No statistical assumptions about data distribution

## ❌ Disadvantages

- **Overfitting**: Prone to creating overly complex trees
- **Instability**: Small data changes can create very different trees
- **Bias**: Biased toward features with more levels
- **Limited Expressiveness**: Axis-aligned splits only
- **High Variance**: Different trees from similar datasets
- **Poor Extrapolation**: Cannot predict beyond training range

## 🎯 When to Use Decision Trees

### ✅ Good For:
- **Interpretability Required**: Need to explain decisions
- **Mixed Data Types**: Both categorical and numerical features
- **Non-linear Patterns**: Complex relationships between features
- **Feature Importance**: Understanding which features matter
- **Quick Baseline**: Fast model to establish performance
- **Rule Extraction**: Converting to business rules

### ❌ Avoid When:
- **Linear Relationships**: Linear models would be simpler
- **High Accuracy Required**: Ensemble methods often better
- **Noisy Data**: Prone to overfitting on noise
- **Small Datasets**: May not have enough data for reliable splits

## 📊 Types of Decision Trees

### Classification Trees
- **Target**: Categorical variables
- **Prediction**: Most common class in leaf
- **Splitting Criteria**: Gini, Entropy, Log Loss
- **Example**: Email spam detection, medical diagnosis

### Regression Trees
- **Target**: Continuous variables
- **Prediction**: Average of values in leaf
- **Splitting Criteria**: MSE, MAE, Friedman MSE
- **Example**: House price prediction, stock forecasting

## 🔧 Hyperparameters

| Parameter | Description | Effect of Increasing | Typical Values |
|-----------|-------------|---------------------|----------------|
| **max_depth** | Maximum tree depth | More complex model, higher overfitting risk | 3-10 |
| **min_samples_split** | Min samples to split node | Simpler model, less overfitting | 2-20 |
| **min_samples_leaf** | Min samples in leaf | Simpler model, smoother boundaries | 1-10 |
| **max_features** | Features considered for best split | More randomness, less overfitting | sqrt(n), log2(n), n |
| **criterion** | Splitting quality measure | Different optimization objectives | gini, entropy, mse |

## 📈 Evaluation and Interpretation

### Model Evaluation
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: MAE, MSE, RMSE, R²
- **Cross-Validation**: K-fold to assess generalization

### Feature Importance
```python
# Scikit-learn provides feature importance
importance = tree.feature_importances_
```
- Based on impurity reduction
- Higher values = more important features
- Sum of all importances = 1.0

### Tree Visualization
- **Text representation**: Rules in if-else format
- **Graphical tree**: Visual tree structure
- **Decision boundaries**: 2D visualization of splits

## 🌳 Tree Pruning

### Purpose
Reduce overfitting by removing parts of the tree that don't improve performance on validation data.

### Types
1. **Pre-pruning** (Early Stopping):
   - Stop growing tree early
   - Use stopping criteria
   
2. **Post-pruning**:
   - Grow full tree, then remove branches
   - Cost complexity pruning (α parameter)

### Cost Complexity Pruning
```
Total Cost = MSE + α × Number of Leaves
```
- α controls trade-off between accuracy and complexity
- Higher α = simpler trees

## 🎯 Implementation Steps

### 1. Data Preparation
```python
# Load data
# Handle missing values (optional for trees)
# Split into train/test
# No scaling required
```

### 2. Model Training
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
model.fit(X_train, y_train)
```

### 3. Model Evaluation
```python
# Make predictions
# Calculate metrics
# Analyze feature importance
# Visualize tree
```

### 4. Hyperparameter Tuning
```python
# Grid search or random search
# Cross-validation
# Pruning parameters
```

## 📊 Diagnostic Techniques

### Learning Curves
- Plot training vs. validation error
- Identify overfitting (large gap)
- Determine if more data would help

### Feature Importance Analysis
- Identify most important features
- Remove irrelevant features
- Understand decision process

### Tree Complexity Analysis
- Plot tree size vs. performance
- Find optimal depth/complexity
- Balance accuracy and interpretability

## 🎯 What You'll Learn

### Theoretical Understanding
- [ ] Information theory concepts (entropy, information gain)
- [ ] Splitting criteria and their properties
- [ ] Pruning theory and practice
- [ ] Bias-variance tradeoff in trees

### Practical Implementation
- [ ] From-scratch implementation of CART algorithm
- [ ] Scikit-learn DecisionTreeClassifier/Regressor
- [ ] Tree visualization techniques
- [ ] Handling categorical variables

### Model Optimization
- [ ] Hyperparameter tuning strategies
- [ ] Cross-validation for tree models
- [ ] Pruning techniques
- [ ] Feature engineering for trees

### Advanced Topics
- [ ] Ensemble methods (Random Forest, Gradient Boosting)
- [ ] Handling imbalanced data with trees
- [ ] Interpretability vs. performance trade-offs
- [ ] Converting trees to business rules

## 🚀 Real-World Applications

- **Healthcare**: Medical diagnosis, treatment recommendations
- **Finance**: Credit scoring, fraud detection
- **Marketing**: Customer segmentation, churn prediction
- **Manufacturing**: Quality control, defect classification
- **HR**: Resume screening, employee retention
- **Retail**: Recommendation systems, inventory management

## 🔗 Connections to Other Algorithms

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential decision trees
- **Rule-based Systems**: Trees can be converted to rules
- **Neural Networks**: Trees can approximate neural network decisions

---

*Decision trees: The foundation of interpretable machine learning! 🌳*
