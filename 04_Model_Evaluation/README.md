# 📊 Model Evaluation & Validation

Model evaluation is crucial for understanding how well your machine learning model performs and whether it will generalize to new, unseen data.

## 🎯 Key Concepts

### 🔄 Validation Strategies
- **Train-Test Split**: Basic data splitting
- **Cross-Validation**: Multiple train-test splits for robust evaluation
- **Time Series Validation**: Respect temporal order in data
- **Stratified Sampling**: Maintain class distribution in splits

### 📈 Performance Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Regression Metrics**: MAE, MSE, RMSE, R²
- **Ranking Metrics**: AUC-ROC, Precision-Recall curves

### ⚖️ Bias-Variance Tradeoff
- **Understanding**: The fundamental tradeoff in ML
- **Overfitting**: High variance, memorizing training data
- **Underfitting**: High bias, too simple model

## 📊 Classification Metrics Deep Dive

### Confusion Matrix
```
                 Predicted
                 No    Yes
Actual    No    TN    FP
          Yes   FN    TP
```

### Core Metrics
| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Balanced datasets |
| **Precision** | TP / (TP + FP) | Cost of false positives is high |
| **Recall (Sensitivity)** | TP / (TP + FN) | Cost of false negatives is high |
| **Specificity** | TN / (TN + FP) | True negative rate important |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance between precision and recall |

### Advanced Classification Metrics

#### ROC-AUC (Receiver Operating Characteristic)
- **What it measures**: Ability to distinguish between classes
- **Range**: 0.5 (random) to 1.0 (perfect)
- **When to use**: Balanced datasets, probability predictions important

#### Precision-Recall AUC
- **What it measures**: Performance on imbalanced datasets
- **When to use**: Imbalanced classes, positive class is rare

#### Cohen's Kappa
- **What it measures**: Agreement accounting for chance
- **Range**: -1 to 1 (0 = random, 1 = perfect)
- **When to use**: Multi-class problems, accounting for class imbalance

## 📈 Regression Metrics Deep Dive

| Metric | Formula | Characteristics | When to Use |
|--------|---------|-----------------|-------------|
| **MAE** | (1/n) Σ\|y - ŷ\| | Robust to outliers | When outliers shouldn't dominate |
| **MSE** | (1/n) Σ(y - ŷ)² | Penalizes large errors heavily | When large errors are particularly bad |
| **RMSE** | √MSE | Same units as target | Most common, interpretable |
| **R²** | 1 - (SSres/SStot) | Proportion of variance explained | Understanding model fit |
| **MAPE** | (100/n) Σ\|y - ŷ\|/y | Percentage error | When relative error matters |

## 🔄 Cross-Validation Strategies

### K-Fold Cross-Validation
```python
# 5-fold CV splits data into 5 parts
# Each part serves as test set once
Fold 1: [Train|Train|Train|Train|Test ]
Fold 2: [Train|Train|Train|Test |Train]
Fold 3: [Train|Train|Test |Train|Train]
Fold 4: [Train|Test |Train|Train|Train]
Fold 5: [Test |Train|Train|Train|Train]
```

### Stratified K-Fold
- Maintains class distribution in each fold
- Essential for imbalanced datasets
- Ensures each fold is representative

### Time Series Cross-Validation
```python
# Respects temporal order
Split 1: [Train     |Test]
Split 2: [Train          |Test]
Split 3: [Train               |Test]
```

### Leave-One-Out (LOO)
- K = number of samples
- Maximum use of data
- High computational cost
- High variance in estimates

## ⚖️ Bias-Variance Tradeoff

### Understanding the Components

#### Bias
- **High Bias (Underfitting)**:
  - Model too simple
  - Doesn't capture underlying patterns
  - Poor performance on both training and test data
  - Examples: Linear regression on non-linear data

#### Variance
- **High Variance (Overfitting)**:
  - Model too complex
  - Memorizes training data noise
  - Good training performance, poor test performance
  - Examples: Deep decision trees, high-degree polynomials

#### Noise
- Irreducible error in the data
- Cannot be reduced by any model
- Sets lower bound on achievable error

### Total Error Decomposition
```
Total Error = Bias² + Variance + Noise
```

### Finding the Sweet Spot
| Model Complexity | Bias | Variance | Total Error |
|------------------|------|----------|-------------|
| Too Simple | High | Low | High |
| Just Right | Medium | Medium | **Minimum** |
| Too Complex | Low | High | High |

## 🎯 Model Selection Strategies

### Hyperparameter Tuning

#### Grid Search
- Exhaustive search over parameter grid
- Guaranteed to find best combination in grid
- Computationally expensive
- Suffers from curse of dimensionality

#### Random Search
- Randomly sample parameter combinations
- More efficient than grid search
- Good for high-dimensional spaces
- May miss optimal combination

#### Bayesian Optimization
- Uses previous results to guide search
- More efficient than random search
- Good for expensive-to-evaluate models
- More complex to implement

### Model Selection Process
1. **Split data** into train/validation/test
2. **Train models** on training set
3. **Tune hyperparameters** using validation set
4. **Select best model** based on validation performance
5. **Final evaluation** on test set (only once!)

## 📊 Dealing with Imbalanced Data

### Detection
- Check class distribution
- Look for skewed metrics (high accuracy, low precision/recall)

### Evaluation Strategies
- Use stratified sampling
- Focus on precision, recall, F1-score
- Use ROC-AUC and Precision-Recall AUC
- Consider class-specific metrics

### Sampling Techniques
- **Undersampling**: Remove majority class samples
- **Oversampling**: Duplicate minority class samples
- **SMOTE**: Generate synthetic minority samples
- **Class weights**: Adjust algorithm to penalize misclassifications differently

## 🎯 Learning Objectives

After completing this section, you should be able to:
- [ ] Choose appropriate metrics for different problem types
- [ ] Implement various cross-validation strategies
- [ ] Understand and diagnose bias-variance tradeoff
- [ ] Perform hyperparameter tuning effectively
- [ ] Handle imbalanced datasets appropriately
- [ ] Avoid common evaluation pitfalls
- [ ] Design robust evaluation pipelines

## ⚠️ Common Pitfalls

### Data Leakage
- Information from future leaking into the model
- Using target variable to create features
- Not respecting temporal order

### Overfitting to Validation Set
- Repeatedly using validation set for model selection
- Solution: Use separate test set for final evaluation

### Inappropriate Metrics
- Using accuracy for imbalanced datasets
- Ignoring business context in metric selection

### Poor Cross-Validation
- Not using stratification for classification
- Ignoring temporal structure in time series
- Too few folds for small datasets

---

*Measure what matters, validate properly! 📊*
