# Airline Passenger Satisfaction Classifier

## Overview
This dataset contains various features related to airline passenger experiences, including service ratings, delay times, travel times, and more. The goal of this project is to predict whether a passenger was **satisfied** or **not satisfied** with their experience.

I built and compared three classification models:
**K-Nearest Neighbors (KNN)**
**Random Forest**
**Logistic Regression**


---
## Data Dictionary
The data was sourced from [Kaggle] https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data


---
## Data Cleaning
- Standardized column names(lowercase, underscores)
- Removed ID and unnamed index columns
- Imputed 310 missing values in 'arrival_delay_in_minutes' using a ratio based method in relation to 'departure_delay_in_minutes'
- Engineered binary features
  - 'is_business_class'
  - 'is_business_travel'
  - 'is_loyal_customer'
- Encoded remaining categorical features


---
## Modeling Approach

I trained and evaluated 3 classifiers
  - **KNN**: Tuned using different values of k
  - **Random Forest**: No scaling required, used 100 trees
  - **Logistic Regression**: Sclaed using 'StandardScaler'

Each model was evaluated using

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix


---
## Evaluation and Results
- Upon analyzing the resulting metrics of the three classifiers I trained—KNN, Logistic Regression, and Random Forest. I found that Random Forest significantly outperformed the other two.

- This came as no surprise, as during data preparation I observed several non-linear relationships between the features and the target variable (whether a passenger was satisfied or not). Random Forest is well-suited for handling such non-linear patterns, as well as multicollinearity, both of which can negatively impact models like Logistic Regression.

- The dataset was also moderately imbalanced, another scenario that Random Forest handles effectively without needing resampling techniques. The model achieved a 96.22% accuracy score, meaning it correctly predicted outcomes for over 96% of passengers in the test set.

- The most important metric in this context was **precision**, particularly for the "neutral or dissatisfied" class, which scored 96%. This was critical because the costliest mistake (Type I error) in this case is incorrectly predicting a passenger was satisfied when they were not. Minimizing false positives ensures that the airline does not develop a false sense of customer satisfaction, which could lead to inaction on service improvements.

