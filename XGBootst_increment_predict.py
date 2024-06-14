import os
import pandas as pd
from xgboost import XGBRegressor, DMatrix, train as xgb_train, Booster
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Training with full data and predicting the final result

# Load data for city yearly averages, county monthly averages, and city monthly averages
city_year_mean_data = pd.read_csv(r'./data/city_year_mean_data.csv')
county_month_mean_data = pd.read_csv(r'./data/county_month_mean_data.csv')
city_month_mean_data = pd.read_csv(r'./data/city_month_mean_data')

# Define features and targets
features = ['height', 'surface', 'gdp', 'pop', 'tmp', 'light', 'co2']
city_year_target = 'electricity'
city_month_target = 'electricity_month'
county_year_target = 'county_electricity'

# City yearly data as initial training data
X_train_1 = city_year_mean_data[features]
y_train_1 = city_year_mean_data[city_year_target]

# County monthly data
X_train_2 = county_month_mean_data[features]
y_train_2 = county_month_mean_data[county_year_target]

# City monthly data
X_train_3 = city_month_mean_data[features]
y_train_3 = city_month_mean_data[city_month_target]

# Set parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [2, 3, 4]
}

# Initialize XGBoost model
xgb = XGBRegressor()

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_1, y_train_1)

# Output the best parameters
print(f'Best parameters: {grid_search.best_params_}')

# Train the best model with the initial training data
xgb_best = grid_search.best_estimator_
xgb_best.fit(X_train_1, y_train_1)
xgb_best.save_model('xgboost_model_predict.json')

# Step 2: Spatial Incremental Learning
loaded_model = Booster()
loaded_model.load_model('xgboost_model_predict.json')

# Combine city yearly data and county monthly data
X_train = pd.concat([X_train_1, X_train_2])
y_train = pd.concat([y_train_1, y_train_2])

# Perform incremental learning
dtrain_new = DMatrix(X_train, label=y_train)
loaded_model = xgb_train(grid_search.best_params_, dtrain_new,
                         num_boost_round=grid_search.best_params_['n_estimators'] + 100, xgb_model=loaded_model)
# Save the updated model
loaded_model.save_model('xgboost_incremental_predict_I.json')

# Step 3: Temporal Incremental Learning
loaded_spa_model = Booster()
loaded_spa_model.load_model('xgboost_incremental_predict_I.json')

# Combine all training data
X_train = pd.concat([X_train_1, X_train_2, X_train_3])
y_train = pd.concat([y_train_1, y_train_2, y_train_3])

# Perform further incremental learning
dtrain_new = DMatrix(X_train, label=y_train)
loaded_model = xgb_train(grid_search.best_params_, dtrain_new,
                         num_boost_round=grid_search.best_params_['n_estimators'] + 100, xgb_model=loaded_spa_model)
loaded_spa_model.save_model('xgboost_incremental_predict_II.json')

# Load the final model for prediction
bst = Booster()
bst.load_model('xgboost_incremental_predict_II.json')

# Define features for prediction
features = ['height', 'surface', 'gdp', 'pop', 'tmp', 'light', 'co2']

# Predict results for all grid monthly data (Sample data only in this .py)
folder_path = r'data/xgboost_stil_predict_input'

# Iterate through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file
        df = pd.read_csv(file_path, encoding='utf_8_sig')
        df.columns = ['no', 'height', 'surface', 'gdp', 'pop', 'tmp', 'light', 'co2', 'county', 'city', 'year', 'month']
        X_test = df[features]
        dtest = DMatrix(X_test)
        y_pred = bst.predict(dtest)
        df['predict'] = y_pred

        # Save the prediction results to a new CSV file
        df.to_csv(r'./xgboost_stil_predict_output' + str(filename), index=False, encoding='utf_8_sig')
