# Insurance Claims Machine Learning Project

This project explores various machine learning tasks using customer data for Sure Tomorrow, an insurance company. The goal is to help the company leverage machine learning to solve key business challenges. The project involves data preprocessing, exploratory data analysis (EDA), and multiple machine learning models to tackle different tasks.

## Tasks

### Task 1: Find Similar Customers
The first task is to identify customers who are similar to a given customer. This helps the company's agents with targeted marketing efforts. The solution involves using k-Nearest Neighbors (kNN) to find similar customers based on a set of features, such as age, income, and family size.

### Task 2: Predict Likelihood of Receiving an Insurance Benefit
For the second task, we build a binary classification model to predict if a new customer is likely to receive an insurance benefit. We evaluate the performance of a kNN classifier and compare it with a dummy model.

### Task 3: Predict Number of Benefits
Predicting the Number of Insurance Benefits a new customer is likely to receive using a linear regression model.

### Task 4: Data Protection
Masking, where we apply an obfuscation technique to protect customer data while preserving the accuracy of the machine learning models.

## Data Preprocessing & Exploration

### Initialization

We begin by importing necessary libraries and setting up the environment. Here's the code to initialize the required packages:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
````

### Load Data

The data is loaded and preprocessed to ensure consistency and validity:

```python
df = pd.read_csv('datasets/insurance_us.csv')
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})
df['age'] = df['age'].astype('int')
```

We also perform a quick check on the data and view its descriptive statistics:

```python
df.info()
df.describe()
```

### Exploratory Data Analysis (EDA)

We create a pair plot to explore the relationships between different customer features. Based on the visualizations, we make initial observations about the data:

* Gender 0 shows a more even income distribution compared to Gender 1.
* The majority of clients are between the ages of 20 and 35.
* Most clients have low or no insurance claims.
* Older individuals are more likely to have insurance claims.

### Task 1: Find Similar Customers

In this task, we use the kNN algorithm to find similar customers. We test both scaled and unscaled data using Euclidean and Manhattan distance metrics. The goal is to understand the impact of data scaling and distance metrics on the similarity calculation.

```python
def get_knn(df, n, k, metric):
    X = df[feature_names]
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(X)
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    df_res = pd.concat([df.iloc[nbrs_indices[0]], pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])], axis=1)
    return df_res
```

### Task 2: Is Customer Likely to Receive Insurance Benefit?

For the second task, we build a KNN classifier to predict whether a customer is likely to receive an insurance benefit. We compare the performance of the classifier using different values of k and measure its performance using the F1-score.

We also implement a dummy model, which predicts insurance benefits based on random probabilities:

```python
def rnd_model_predict(P, size, seed=42):
    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)
```

### Model Evaluation

We split the dataset into training and testing sets and evaluate the models using metrics like F1-score and confusion matrix.

```python
explicit_train_df = df.sample(frac=0.7, random_state=state)
explicit_test_df = df.drop(explicit_train_df.index)

explicit_train_scaled = scaleData(explicit_train_df)
explicit_test_scaled = scaleData(explicit_test_df)

X_train = explicit_train_df[feature_names]
y_train = explicit_train_df['insurance_benefits_received']
```

## Results and Observations

* **Task 1 (Find Similar Customers)**: Scaling the data significantly improves the accuracy of the kNN algorithm, especially in terms of finding more reasonable nearest neighbors.
* **Task 2 (Predict Insurance Benefit)**: The kNN classifier performs better when the data is scaled. The dummy model provides a baseline for comparison.

## Conclusion

This project demonstrates the power of machine learning techniques such as kNN and linear regression for predicting insurance claims and customer behavior. By incorporating data scaling and proper evaluation metrics, we can improve model performance and offer valuable insights for the business.

### Task 3: Predict Number of Insurance Benefits

In this task, we are tasked with predicting the number of insurance benefits that a new customer is likely to receive using a linear regression model.

## Code Explanation

We begin by reading the insurance dataset and renaming the columns for easier manipulation. We also create a binary target variable, `insurance_benefits_received`, indicating whether the customer received any insurance benefits.

```python
df = pd.read_csv('datasets/insurance_us.csv')
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})
df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int)
```

Next, we select relevant features and split the data into training and testing sets. We apply scaling to the features to normalize them for the linear regression model.

```python
features = df[feature_names]
target = df['insurance_benefits_received']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(features, target, test_size=0.3, random_state=state)
X_train_scaled = scaleData(X_train_new)
X_test_scaled = scaleData(X_test_new)
```

We then train a custom linear regression model using the training data and evaluate its performance on the test data.

```python
lr = MyLinearRegression()

lr.fit(X_train_new, y_train_new)
print(lr.weights)

y_test_new_pred = lr.predict(X_test_new)
eval_regressor(y_test_new, y_test_new_pred)

lr.fit(X_train_scaled, y_train_new)
print(lr.weights)

y_test_new_pred_scaled = lr.predict(X_test_scaled)
eval_regressor(y_test_new, y_test_new_pred_scaled)
```

## Analysis

The analysis shows that the RMSE (Root Mean Squared Error) and R² (Coefficient of Determination) are similar for both the scaled and original datasets, indicating that the linear regression model performs consistently well regardless of whether the data is scaled.

### Analysis of RMSE and R² on Scaled and Original Dataset

It appears that, even with the class imbalance, the RMSE and R² values between the scaled and original dataset are similar. The weights, beyond gender, differ quite substantially, but both have a low RMSE of 0.23 and R² of 0.44.

### Task 4: Data Protection - Masking

To protect customer data, we apply a technique known as data obfuscation, where we transform the data by multiplying the feature matrix by an invertible matrix 𝑃. The challenge is to ensure that this transformation does not affect the performance of the machine learning model.

## Code Explanation

We select columns containing personal information and convert them into a NumPy array. We then generate a random invertible matrix 𝑃 and apply it to obfuscate the data.

```python
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]
X = df_pn.to_numpy()

# Generating a random invertible matrix P
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

# Check if P is invertible
det_P = np.linalg.det(P)
print(f"Determinant of P: {det_P}")

if det_P != 0:
    print("Matrix P is invertible.")
else:
    print("Matrix P is not invertible.")
```

If the matrix 𝑃 is invertible (determinant ≠ 0), we proceed to transform the data by multiplying the original matrix 𝑋 with 𝑃.

```python
X_prime = X.dot(P)
```

We can then recover the original data by applying the inverse of 𝑃, which helps prove that the transformation is reversible.

```python
P_inv = np.linalg.inv(P)
X_recovered = X_prime.dot(P_inv)
```

## Displaying the original, transformed, and recovered matrices

```python
display('Original First Five Columns:')
display(df.head())

display('Transformed Matrix (First 5):')
display(X_prime[:5])

display('Recovered Matrix (First 5):')
display(X_recovered[:5])
```

## Analysis of Recovered Data

The analysis shows that, while the recovered data is not exactly the same as the original, the discrepancies are due to floating point inaccuracies inherent in matrix calculations.

## Proof of Linear Regression with Obfuscated Data

We mathematically prove that the obfuscation does not affect the linear regression model’s predicted values by showing that the predicted values with the obfuscated data are identical to those with the original data.

The relationship between the original weights 𝑤 and the obfuscated weights 𝑤𝑃 is derived as follows:

```
𝑤𝑃 = 𝑃⁻¹𝑤
```

This leads to the conclusion that the predicted values using 𝑤𝑃 are identical to those predicted using the original weights 𝑤:

```
𝑦̂𝑃 = 𝑦̂
```

### Computational Proof

We validate this theoretical analysis by running linear regression on both the original and obfuscated data and comparing the performance using RMSE and 𝑅².

```python
lr.fit(X_train, y_train)
y_test_pred = lr.predict(X_test)

lr.fit(X_tr_obs, y_train)
y_test_pred_obs = lr.predict(X_test_obs)

print('Original Data: ')
eval_regressor(y_test, y_test_pred)

print(f'\nObfuscation Data: ')
eval_regressor(y_test, y_test_pred_obs)
```

## Conclusion

The results show no significant difference in performance (RMSE and 𝑅²) between the original and obfuscated datasets, confirming that the obfuscation technique preserves the predictive accuracy of the linear regression model.

### Final Conclusion

By applying data obfuscation techniques, we can successfully protect customer data without sacrificing the performance of the linear regression model. This approach can be extended to larger datasets, making it a viable solution for maintaining data privacy while using machine learning models.

## Technologies Used

* Python
* Pandas
* Numpy
* Scikit-learn
* Seaborn

## Installation

To run this project locally, clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/insurance-claims-ml.git
cd insurance-claims-ml
pip
```


Necessary libraries will be installed at initialization when running the notebook.

```
```

You can now easily copy this markdown content into your file!
