# Titanic Survival Prediction
This repository contains a machine learning model for predicting the survival of passengers aboard the Titanic. The model uses logistic regression and several feature engineering techniques to predict whether a passenger survived based on various attributes.

Dataset  
The dataset used for this project is from the Titanic: Machine Learning from Disaster competition on Kaggle (https://www.kaggle.com/competitions/titanic/overview). It contains information about the passengers, including features like Age, Fare, Pclass, Sex, Embarked, etc.

## Data Preprocessing:

### Handling Missing Values:
* Age: Filled missing values with the median age.
* Cabin: Converted to a binary feature (0 for missing, 1 for present).
* Embarked: Imputed missing values with the most frequent value.
* Fare: Filled missing values using the median fare per Pclass.

### Feature Engineering:
* IsChild: Created a binary feature for passengers under 15 years old.
* Family Size: Combined SibSp and Parch to create a new Family_size feature and categorized into "Single", "SmallFamily", and "LargeFamily".
* Fare Binned: Created categorical bins for Fare.
* Age Binned: Created age groups: "0-6", "7-15", "16-35", "36-50", "51-65", "66-80".
* Ticket Counts: Counted occurrences of each ticket number.

### Encoding and Scaling:
* Sex: Converted to binary encoding (0 = male, 1 = female).
* Embarked, Family Category, Age Binned, and Fare Binned: Applied One-Hot Encoding.
* StandardScaler: Standardized numerical features (Age, Fare, and binned features).

## Modeling:
1. Used Logistic Regression as the baseline model.
2. Applied RandomizedSearchCV for hyperparameter tuning to optimize model parameters such as C and solver.
3. Addressed class imbalance using techniques like SMOTE.

## Evaluation:
The final model achieved an accuracy of ~81.88% on the test set, with good precision and recall, indicating balanced performance across the classes.  
* Accuracy: 81.88%
* Precision: 83.14%
* Recall: 79.96%

* Confusion Matrix:  
[460  89]  
[110 439]  

## Libraries Used
* pandas: For data manipulation and preprocessing.
* scikit-learn: For machine learning models, evaluation, and hyperparameter tuning.
* imbalanced-learn: For handling class imbalance using SMOTE.
* matplotlib & seaborn: For visualization (if applicable).

## Results
After preprocessing the data, training the model, and performing hyperparameter tuning, the final model achieved the following results:


## Future Improvements
* Explore additional algorithms such as Random Forest or XGBoost for potentially better performance.
* Further tune hyperparameters and experiment with more complex feature engineering techniques.
* Try different techniques for dealing with missing values (e.g., KNN imputation for Fare or Age).

