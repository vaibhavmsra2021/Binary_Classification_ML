# Classification of Rice Grain Species using Morphological Features

## Project Overview
This project focuses on the classification of two rice species—Cammeo and Osmancik—using morphological features extracted from rice grain images. The goal is to implement and evaluate multiple machine learning models to achieve high classification accuracy.

## Problem Statement
Rice classification plays a crucial role in quality control and variety identification in agriculture and the food industry. This project aims to:
- Identify significant morphological features distinguishing the two species.
- Implement machine learning techniques for classification.
- Evaluate and compare the performance of various classification models.

## Objectives
- Preprocess the dataset and extract meaningful features.
- Implement and evaluate multiple classification models.
- Optimize models through hyperparameter tuning.
- Analyze feature importance.
- Visualize model performance using confusion matrices and ROC-AUC curves.

## Dataset and Feature Description
The dataset includes morphological features:
- **Area**: Number of pixels within the boundaries of the rice grain.
- **Perimeter**: Circumference of the rice grain boundary.
- **Major Axis Length**: Longest possible line across the rice grain.
- **Minor Axis Length**: Shortest possible line across the rice grain.
- **Eccentricity**: Measure of the rice grain’s roundness.
- **Convex Area**: Pixel count of the smallest convex shell around the grain.
- **Extent**: Ratio of the grain region to the bounding box area.

## Machine Learning Models Used
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Decision Tree**
4. **Random Forest**
5. **Gradient Boosting**
6. **K-Nearest Neighbors (KNN)**
7. **Neural Networks**

## Evaluation Metrics
- **Accuracy**: Percentage of correct predictions.
- **Confusion Matrix**: Breakdown of prediction results.
- **ROC-AUC**: Performance measure using true positive and false positive rates.

## Solution Methodology
### 1. Dataset Preprocessing
- Loaded and cleaned the dataset (in .arff format).
- Encoded the target variable (Class) into numeric values (0 for Cammeo, 1 for Osmancik).
- Scaled feature values using **StandardScaler**.

### 2. Model Implementation
- Implemented models using **scikit-learn**.
- Evaluated models on an **80-20 train-test split**.

### 3. Feature Importance Analysis
- Extracted feature importance scores using **Random Forest**.
- Applied **Recursive Feature Elimination (RFE)** for feature selection.

### 4. Hyperparameter Tuning
- Optimized hyperparameters using **GridSearchCV**.
- Used **cross-validation** to evaluate parameter combinations.

### 5. Visualization
- Plotted confusion matrices, ROC curves, and feature importance charts.

## Simulation Results
### Feature Importance
The most significant features were:
1. **Major Axis Length**
2. **Perimeter**
3. **Area**
4. **Convex Area**
5. **Eccentricity**

### Model Performance (After Hyperparameter Tuning)
| Model                  | Accuracy  | ROC-AUC  |
|------------------------|----------|---------|
| Logistic Regression    | 0.9278   | 0.92    |
| Support Vector Machine | 0.9318   | 0.93    |
| Decision Tree         | 0.8911   | 0.89    |
| Random Forest         | 0.9265   | 0.95    |
| Gradient Boosting     | 0.9304   | 0.94    |
| KNN                   | 0.9186   | 0.95    |
| Neural Network        | 0.9304   | 0.95    |

## Conclusion
- **Support Vector Machine (SVM) achieved the highest accuracy (93.7%) and ROC-AUC (0.97)**.
- **Area, Perimeter, and Major Axis Length were the most important features**.
- Hyperparameter tuning slightly improved model accuracy.
- Visualization using confusion matrices and ROC curves provided deeper insights into model performance.

## References
1. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, 2011.
2. Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*.
3. Dataset Source: Rice (Cammeo and Osmancik) [Dataset]. (2019). UCI Machine Learning Repository.

## Supplementary Material
- Full Python code and dataset available in supplementary files.

## Contribution
- **Vaibhav Mishra (CH21B038)** – 100% of the contributions.

---
This README provides an overview of the project, methodologies, and results. For implementation details, refer to the full project report and code files.

