import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import catboost
import pickle
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/house_kg_10K_ads.csv')

# Feature Engineering
df.loc[df['rooms'] == "свободная планировка", 'rooms'] = 0
df.loc[df['rooms'] == "6 и более", 'rooms'] = 6
df["rooms"] = df["rooms"].astype(int)
#df['is_top_floor'] = (df['floor'] == df['floors']) & (df['floor'] != 1)
#df['is_bottom_floor'] = (df['floor'] == 1) & (df['floor'] != df['floors'])
#df['max_price_micro_district'] = df.groupby('micro_district')['price'].transform('max')
#f['max_price_micro_district'].fillna(0, inplace=True)

# Replace nan values in 'micro_district' with a placeholder
df['micro_district'] = df['micro_district'].fillna('Unknown')

# Encoding categorical data with separate LabelEncoders
district_encoder = LabelEncoder()
micro_district_encoder = LabelEncoder()
building_type_encoder = LabelEncoder()
condition_encoder = LabelEncoder()

df['district_encoded'] = district_encoder.fit_transform(df['district'])
df['micro_district_encoded'] = micro_district_encoder.fit_transform(df['micro_district'])
df['building_type_encoded'] = building_type_encoder.fit_transform(df['building_type'])
df['condition_encoded'] = condition_encoder.fit_transform(df['condition'])

# Save the encoders
pickle.dump(district_encoder, open('district_encoder.pkl', 'wb'))
pickle.dump(micro_district_encoder, open('micro_district_encoder.pkl', 'wb'))
pickle.dump(building_type_encoder, open('building_type_encoder.pkl', 'wb'))
pickle.dump(condition_encoder, open('condition_encoder.pkl', 'wb'))

# Define independent and dependent variables
independent_variables = ["square", "rooms", "floors", "floor", "date_year", 
                         "district_encoded", "micro_district_encoded", 
                         "building_type_encoded", "condition_encoded"]
dependent_variable = "price"

# Set up X and y based on dependent and independent variables
X = df[independent_variables]
y = df[dependent_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Linear Regression Model
regressor_model = LinearRegression()
regressor_model.fit(X_train, y_train)
y_pred = regressor_model.predict(X_test)
print("Linear Regression R-squared:", r2_score(y_test, y_pred))
print("Linear Regression Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
with open('Linear_regressor_model.pkl', 'wb') as file:
    pickle.dump(regressor_model, file)
    print("Linear Regression model saved successfully.")

# Decision Tree Regressor
regressor2 = DecisionTreeRegressor(random_state=42)
regressor2.fit(X_train, y_train)
y_pred = regressor2.predict(X_test)
print("Decision Tree Regressor Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Decision Tree Regressor R Squared Value:", r2_score(y_test, y_pred))
with open('DecisionTreeRegressor_model.pkl', 'wb') as file:
    pickle.dump(regressor2, file)
    print("Decision Tree model saved successfully.")

# Random Forest Regressor
regressor3 = RandomForestRegressor(n_estimators=100, random_state=42)
regressor3.fit(X_train, y_train)
y_pred = regressor3.predict(X_test)
print("Random Forest Regressor Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Random Forest Regressor R Squared Value:", r2_score(y_test, y_pred))
with open('RandomForestRegressor_model.pkl', 'wb') as file:
    pickle.dump(regressor3, file)
    print("Random Forest model saved successfully.")



# LightGBM Regressor
# regressor_lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.1, max_depth=10)
# regressor_lgbm.fit(X_train, y_train)
# y_pred = regressor_lgbm.predict(X_test)
# print("LightGBM R2 Score:", mean_squared_error(y_test, y_pred))
# print("LightGBM MSE:", r2_score(y_test, y_pred))
# with open('lgbm_model.pkl', 'wb') as file:
#     pickle.dump(regressor3, file)
#     print("lgbm model saved successfully.")    
    
    
#a CatBoost Regressor
regressor_catboost = CatBoostRegressor(n_estimators=100, random_state=42)
regressor_catboost.fit(X_train, y_train)
y_pred_catboost = regressor_catboost.predict(X_test)
mse_catboost = mean_squared_error(y_test, y_pred_catboost)
r2_catboost = r2_score(y_test, y_pred_catboost)
print("CatBoost Regressor Mean Squared Error:", mse_catboost)
print("CatBoost Regressor R-squared Value:", r2_catboost)

# Save the CatBoost model to a file
with open('CatBoostRegressor_model.pkl', 'wb') as file:
    pickle.dump(regressor_catboost, file)
    print("CatBoost model saved successfully.")