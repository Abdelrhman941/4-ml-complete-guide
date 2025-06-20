
### Types of Ensemble Methods

#### 1. Bagging (Bootstrap Aggregating)
**Strategy:** Train multiple models on different subsets of data

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Random Forest** | Bagging + Random features | Reduces overfitting, fast | Less interpretable |
| **Extra Trees** | Extremely randomized trees | Even faster training | Higher variance |

**How it works:**
```
Dataset → Bootstrap Sample 1 → Model 1 ┐
        → Bootstrap Sample 2 → Model 2 ├→ Average/Vote → Final Prediction
        → Bootstrap Sample n → Model n ┘
```

#### 2. Boosting
**Strategy:** Train models sequentially, each correcting previous errors

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **AdaBoost** | Adaptive boosting | Good performance, interpretable | Sensitive to noise |
| **Gradient Boosting** | Fits residuals iteratively | Excellent performance | Prone to overfitting |
| **XGBoost** | Optimized gradient boosting | State-of-the-art performance | Many hyperparameters |
| **LightGBM** | Fast gradient boosting | Very fast, efficient | Requires tuning |
| **CatBoost** | Handles categorical features | Automatic cat. handling | Less mature |

**How it works:**
```
Model 1 → Errors 1 → Model 2 → Errors 2 → Model 3 → ... → Final Model
```

#### 3. Stacking
**Strategy:** Use a meta-model to combine predictions from base models

**How it works:**
```
Base Model 1 ┐
Base Model 2 ├→ Meta-Model → Final Prediction
Base Model 3 ┘
```

#### 4. Voting
**Strategy:** Combine predictions through voting or averaging

- **Hard Voting**: Majority vote for classification
- **Soft Voting**: Average predicted probabilities


## 📊 Performance Comparison

### Ensemble vs Individual Models
```
Individual Model:     ████████░░ 80% accuracy
Random Forest:        ██████████ 85% accuracy  (+5%)
Gradient Boosting:    ███████████ 88% accuracy (+8%)
Stacked Ensemble:     ████████████ 90% accuracy (+10%)
```