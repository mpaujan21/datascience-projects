import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# Dataset: https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes
df_bmw = pd.read_csv('bmw.csv')

# Change 'year' column
df_bmw['car_age'] = 2020 - df_bmw['year']
df_bmw.drop(columns='year', inplace=True)

# One Hot Encoding
df = pd.get_dummies(df_bmw)

# Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Split to Train&Test Data
from sklearn.model_selection import train_test_split

X = df_scaled.drop(columns='price')
y = df_scaled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# Linear Model with SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

column_names = df_scaled.drop(columns='price').columns
no_of_features = []
r2_train = []
r2_test = []

for k in range(3,38,2):
    selector = SelectKBest(f_regression, k=k)
    X_train_transformed = selector.fit_transform(X_train, y_train)
    X_test_transformed = selector.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r2_train.append(regressor.score(X_train_transformed, y_train))
    r2_test.append(regressor.score(X_test_transformed, y_test))

selector = SelectKBest(f_regression, k=31)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

def regression_model(model):
    """
    Fit the regression model passed, then it will return the regressor object and it's score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score

model_performance = pd.DataFrame(columns=['Model', 'Score'])
linear_model = [LinearRegression(), Ridge(), SVR(), RandomForestRegressor()]

for model in linear_model:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({'Feature':'Linear', 'Model': model, 'Score': score}, ignore_index=True)

model_performance = model_performance.sort_values('Score', ascending=False)

# Backward Elimination
import statsmodels.api as sm

regressor = sm.OLS(y_train, X_train).fit()
print(regressor.summary())
X_train_dropped = X_train.copy()

while True:
    if max(regressor.pvalues) > 0.05:
        drop_variable = regressor.pvalues[regressor.pvalues == max(regressor.pvalues)]
        print('Dropping ', drop_variable.index[0] + 'with p-value of ', str(drop_variable[0]))
        print('Running regression again...\n')
        X_train_dropped = X_train_dropped.drop(columns=drop_variable.index[0])
        regressor = sm.OLS(y_train, X_train_dropped).fit()
    else:
        print('All p-values less than 0.05')
        break

model_performance = model_performance.append({'Feature': 'Linear with Backward Elimination', 'Model': 'LinearRegression()', 'Score': regressor.rsquared}, ignore_index=True)

# Polynomial Model with SelectKBest
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
X_train_poly.shape

no_of_features = []
r2_train = []
for k in range(10,741,5):
    selector = SelectKBest(f_regression, k=k)
    X_train_transformed = selector.fit_transform(X_train_poly, y_train)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r2_train.append(regressor.score(X_train_transformed, y_train))

selector = SelectKBest(f_regression, k=700)
X_train_transformed = selector.fit_transform(X_train_poly, y_train)
X_test_transformed = selector.transform(X_test_poly)

linear_model = [LinearRegression(), Ridge(), SVR(), RandomForestRegressor()]

for model in linear_model:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({'Feature':'Polynomial', 'Model': model, 'Score': score}, ignore_index=True)

model_performance = model_performance.sort_values('Score', ascending=False)
