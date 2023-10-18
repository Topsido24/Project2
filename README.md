**Week 2 Project: Predicting Booking Prices on Airbnb in Asterdam**
Airbnb is the most well-known company for short-term housing rentals. We want to see which of two models is more effective in prdicting the price of Airbnb. 


**Step 1: Defining the Events**
Airbnb offers online marketplace for short-and long-term homestays and experiences. In trying to predict the booking prices on Airbnb, we will be considering two models which are; the linear regression model and the random forest regression model. We want to see which of the models is perform better. We will be using data gotten from Kaggle and taken from the Data Card of "Airbnb Prices in European Cities" and we will only be considering Asterdam.
We wil need some framework/libraries to make our prediction easy.
```python

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

```

**Step 2: Data Collection**
Some of the factors we will be considering for this prediction are; the total price of the Airbnb listing (realSum), the type of room being offered (room_type), whether the room is shared or not (room_shared), whether the room is private or not (room_private), the maximum number of people that can stay in the room (person_capacity), the cleanliness rating of the listing, the overall guest satisfaction of the listing and many more.
We will gret our data from the Kaggle website from the avsailable data on it. We will save in csv than import into our jupyter notebook using the syntax;
```python

df1 = pd.read_csv("/kaggle/input/airbnb-prices-in-european-cities/amsterdam_weekdays.csv")
df2 = pd.read_csv("/kaggle/input/airbnb-prices-in-european-cities/amsterdam_weekends.csv")


# Concatenate the DataFrames
amsterdam_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
amsterdam_df.to_csv('amsterdam_data.csv', index=False)

amsterdam_df                       #display the dataset

```


**Step 3: Data Preprocessing**
After importing the collected data, we will process the data thoroughly to work on some of the irregularities. These irregularities must be handle to get what we want. The things we want to correct are;
Knowing details about the dataset
Handling missing data: We will remove rows with missing data, outliers, and duplicate data.
Normalizing numerical features like total price of the Airbnb listing.
Identifying and filtering categorical columns

```python

amsterdam_df.shape

amsterdam_df.columns

amsterdam_df.info()

amsterdam_df.describe()

#Checking distributions of the various features in the dataset

index= ["room_shared","realSum", "room_type","room_private","person_capacity","host_is_superhost", "multi", "biz", "cleanliness_rating",
        "guest_satisfaction_overall", "bedrooms", "dist", "metro_dist", "attr_index_norm", "attr_index", "rest_index", "rest_index_norm", "lng", 
        "lat"]

for i in index:
    
    print(amsterdam_df[i].value_counts(), "\n")
    print("---------------------------------------------------------------")

data.dropna(inplace=True)		#handling missing data
data.drop_duplicates(inplace=True)		#removing duplicate data
scaler = StandardScaler()
data[‘realSum’] = scaler.fit_transform(data[‘realSum’].values.reshape(-1, 1))		#normalize numerical features (e.g total price of the Airbnb listing)

# Identify and filter categorical columns
categorical_cols = [col for col in amsterdam_df.columns if amsterdam_df[col].dtype == 'object']

# Calculate the number of rows needed based on the number of categorical columns
num_col = len(categorical_cols)
num_row = (num_col + 2) // 3  # Calculate the number of rows needed

plt.figure(figsize=(15, 5 * num_row))  # Adjust the figure size based on the number of rows

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(num_row, 3, i)  # 3 columns per row
    sns.countplot(data=amsterdam_df, x=col)
    plt.title(f'Count Plot of {col}')

plt.tight_layout()
plt.show()

```

**Step 4: Feature Engineering and Selection**
Now, we need to identify and select critical features for predicting booking prices and we will also drop columns that are not important for our analysis. It is also here will be able to add any features that will make our model better. We will also transform our categorical data using Label Encoding method. We will also check for collinearity between the target variable and the independent variables to see which of the features we need to drop so that it doesn't affect our model. Another thing we want to achieve here is to train our model.

```python

unnamed = amsterdam_df['Unnamed: 0']
attr_index = amsterdam_df['attr_index']
attr_index_norm = amsterdam_df['attr_index_norm']
rest_index = amsterdam_df['rest_index']
rest_index_norm = amsterdam_df['rest_index_norm']

# Drop the specified columns from the DataFrame
columns_to_drop = ['Unnamed: 0', 'attr_index', 'attr_index_norm', 'rest_index', 'rest_index_norm']
amsterdam_df = amsterdam_df.drop(columns_to_drop, axis=1)

# Display the updated DataFrame
amsterdam_df.head()

amsterdam_df = pd.get_dummies(amsterdam_df)
amsterdam_df.head()

le = LabelEncoder()

for col in amsterdam_df.columns:
    if amsterdam_df[col].dtype == 'bool':
        print(f"Column '{col}' is boolean and will be converted to binary using LabelEncoder.")
        
        # Fit and transform the LabelEncoder on the column
        amsterdam_df[col] = le.fit_transform(amsterdam_df[col])

amsterdam_df.head()

y = amsterdam_df['realSum'] #Target variable / Dependent variable

X = amsterdam_df.drop('realSum', axis =1) # Independent variable

correlation = X.corrwith(y)
print(correlation)

plt.figure(figsize=(12, 6))
sns.barplot(x=correlation.values, y=correlation.index, palette="viridis")
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.title('Correlation between Features and Booking Prices')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

corr_matrix = X.corr()

# Set the font size for labels
plt.rc('font', size=8)

# Create a heatmap plot of the correlation matrix
plt.figure(figsize=(12, 12))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Correlation Matrix between features')
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
plt.tight_layout()
plt.show()

print(corr_matrix)

from statsmodels.stats.outliers_influence import variance_inflation_factor #Handling multicollinearity
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

X = X.drop(['room_shared','room_private', 'room_type_Private room','cleanliness_rating', 'room_type_Shared room'], axis=1)
X.head()

X = amsterdam_df['person_capacity'].values
y = amsterdam_df['realSum'].values

print(X)
print(y)

X = X.reshape(-1,1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=100)

print(f'X_train shape{X_train}')
print(f'X_test shape{X_test}')
print(f'y_train shape{y_train}')
print(f'y_test shape{y_test}')

plt.scatter(X_train, y_train)
plt.xlabel('Person Capacity')
plt.ylabel('Price of Airbnb')
plt.title('Amsterdam Airbnb Training Data')
plt.show()

```
**Step 5: Model Selection**
We will be using the Linear Regression model and the Random Forest Regression model to predict the booking price. We want to see which of the model will perform better.

```python
#For Linear Regression

lm = LinearRegression()
lm.fit(X_train, y_train)
y_predict = lm.predict(X_test)

#For Random Forest Regression

from sklearn.metrics import mean_squared_error, r2_score

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)


```


**Step 6: Model Evaluation**
We will now check which of the models performed better among the two models.

```python

#For Linear Regression 

print(f'Train Accuracy {round(lm.score(X_train, y_train)* 100,2)}%')
print(f'Test Accuracy {round(lm.score(X_test, y_test)* 100,2)}%')

#For Random Forest Regression

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

```

**Communication of results**
After checking both models, the Linear Regression model gave us a 29.39% Train Accuracy and a 23.28% Test Accuracy while the Random Forest Regression gave us a 0.9319 Mean Squared Error and a 0.2688 R-squared scores. Based on this, the Random Forest Regression model performed better because of the prediction accuracy and the goodness of fit.


