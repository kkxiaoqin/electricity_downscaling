import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Load data
city_year_mean_data = pd.read_csv(r'./city_density_feature_year_sample.csv')
county_month_mean_data = pd.read_csv(r'./county_density_feature_year_sample.csv')
city_month_mean_data = pd.read_csv(r'./city_density_feature_month_sample.csv')

# Define features and targets
features = ['height', 'surface', 'gdp', 'pop', 'tmp', 'light']
city_year_target = 'electricity'
city_month_target = 'electricity_month'
county_year_target = 'county_electricity'

# Prepare training data
X_train = city_year_mean_data[features]
y_train = city_year_mean_data[city_year_target]

# Parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [2, 3, 4]
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize and train model with grid search
xgb = XGBRegressor()
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf, scoring='r2')
grid_search.fit(X_train, y_train)

# Train best model
xgb_best = grid_search.best_estimator_
xgb_best.fit(X_train, y_train)
xgb_best.save_model('xgboost_model.json')

# Prepare data for cross-validation
def prepare_data(df, target):
    data_train, label_train, data_test, label_test, test_index = [], [], [], [], []
    for train_idx, test_idx in kf.split(df):
        data_train.append(df.iloc[train_idx][features])
        label_train.append(df.iloc[train_idx][target])
        data_test.append(df.iloc[test_idx][features])
        label_test.append(df.iloc[test_idx][target])
        test_index.append(test_idx)
    return data_train, label_train, data_test, label_test, test_index

# City data and county data used for testing
data_train_city, label_train_city, data_test_city, label_test_city, city_test_index = prepare_data(city_month_mean_data, city_month_target)
data_train_county, label_train_county, data_test_county, label_test_county, county_test_index = prepare_data(county_month_mean_data, county_year_target)

# Evaluate model
r2_scores_spatial_all, r2_scores_temporal_all = [], []
rmse_scores_spatial_all, rmse_scores_temporal_all = [], []

for i in range(5):
    # County prediction
    county_predict = xgb_best.predict(data_test_county[i])
    county_true = county_month_mean_data.iloc[county_test_index[i]][county_year_target].values
    r2_scores_spatial_all.append(r2_score(county_true, county_predict))
    rmse_scores_spatial_all.append(mean_squared_error(county_true, county_predict, squared=False))

    # City prediction
    city_predict = xgb_best.predict(data_test_city[i])
    city_true = city_month_mean_data.iloc[city_test_index[i]][city_month_target].values
    r2_scores_temporal_all.append(r2_score(city_true, city_predict))
    rmse_scores_temporal_all.append(mean_squared_error(city_true, city_predict, squared=False))

# Calculate and print average scores
r2_avg_spatial = np.mean(r2_scores_spatial_all)
r2_avg_temporal = np.mean(r2_scores_temporal_all)
f1_score_r2 = (2 * r2_avg_temporal * r2_avg_spatial) / (r2_avg_temporal + r2_avg_spatial)
rmse_avg_spatial = np.mean(rmse_scores_spatial_all)
rmse_avg_temporal = np.mean(rmse_scores_temporal_all)
f1_score_rmse = (2 * rmse_avg_spatial * rmse_avg_temporal) / (rmse_avg_spatial + rmse_avg_temporal)

print(f'Average r2 score for spatial: {r2_avg_spatial}')
print(f'Average r2 score for temporal: {r2_avg_temporal}')
print(f'F1 score for r2: {f1_score_r2}')
print(f'Average RMSE score for spatial: {rmse_avg_spatial}')
print(f'Average RMSE score for temporal: {rmse_avg_temporal}')
print(f'F1 score for RMSE: {f1_score_rmse}')
