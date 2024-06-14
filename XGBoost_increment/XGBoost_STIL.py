import pandas as pd
import numpy as np
from xgboost import XGBRegressor, DMatrix, train as xgb_train, Booster
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

# 读取数据
city_year_mean_data = pd.read_csv(r'./city_density_feature_year_sample.csv')
county_month_mean_data = pd.read_csv(r'./county_density_feature_year_sample.csv')
city_month_mean_data = pd.read_csv(r'./city_density_feature_month_sample.csv')

# 定义自变量和因变量
features = ['height', 'surface', 'gdp', 'pop', 'tmp', 'light']
city_year_target = 'electricity'
city_month_target = 'electricity_month'
county_year_target = 'county_electricity'

# 地级市年度数据当作训练数据
X_train_1 = city_year_mean_data[features]
y_train_1 = city_year_mean_data[city_year_target]

# 设置参数范围
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [2, 3, 4]
}

# 初始化五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化模型
xgb = XGBRegressor()

# 使用网格搜索进行参数寻优
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_1, y_train_1)

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare data for cross-validation
def prepare_data(df, target, kf):
    data_train, label_train, data_test, label_test, test_index = [], [], [], [], []
    for train_idx, test_idx in kf.split(df):
        data_train.append(df.iloc[train_idx][features])
        label_train.append(df.iloc[train_idx][target])
        data_test.append(df.iloc[test_idx][features])
        label_test.append(df.iloc[test_idx][target])
        test_index.append(test_idx)
    return data_train, label_train, data_test, label_test, test_index

data_train_city, label_train_city, data_test_city, label_test_city, city_test_index = prepare_data(city_month_mean_data, city_month_target, kf)
data_train_county, label_train_county, data_test_county, label_test_county, county_test_index = prepare_data(county_month_mean_data, county_year_target, kf)

# Initialize lists to store results
r2_scores_spatial_all = []
r2_scores_temporal_all = []
rmse_scores_temporal_all = []
rmse_scores_spatial_all = []

# Perform incremental learning and evaluation
for i in range(5):
    X_train_3 = data_train_city[i]
    y_train_3 = label_train_city[i]
    X_train_2 = data_train_county[i]
    y_train_2 = label_train_county[i]
    X_train = pd.concat([X_train_1, X_train_2, X_train_3])
    y_train = pd.concat([y_train_1, y_train_2, y_train_3])

    # Load the pre-trained model
    loaded_spa_model = Booster()
    loaded_spa_model.load_model(f'xgboost_STL_{i}.json')

    # Perform incremental learning on the loaded model
    dtrain_new = DMatrix(X_train, label=y_train)
    loaded_spa_model = xgb_train(grid_search.best_params_, dtrain_new,
                                 num_boost_round=grid_search.best_params_['n_estimators'] + 100, xgb_model=loaded_spa_model)

    # Spatial prediction (county level)
    county_predict = loaded_spa_model.predict(data_test_county[i])
    county_true = county_month_mean_data.iloc[county_test_index[i]][county_year_target].values
    r2_scores_spatial_all.append(r2_score(county_true, county_predict))
    rmse_scores_spatial_all.append(mean_squared_error(county_true, county_predict, squared=False))

    # Temporal prediction (city level)
    city_predict = loaded_spa_model.predict(data_test_city[i])
    city_true = city_month_mean_data.iloc[city_test_index[i]][city_month_target].values
    r2_scores_spatial_all.append(r2_score(city_true, city_predict))
    rmse_scores_spatial_all.append(mean_squared_error(city_true, city_predict, squared=False))

# Calculate averages and F1 scores
r2_avg_spatial = np.mean(r2_scores_spatial_all)
r2_avg_temporal = np.mean(r2_scores_temporal_all)
f1_score_for_r2 = (2 * r2_avg_temporal * r2_avg_spatial) / (r2_avg_temporal + r2_avg_spatial)
rmse_avg_spatial = np.mean(rmse_scores_spatial_all)
rmse_avg_temporal = np.mean(rmse_scores_temporal_all)
f1_score_for_rmse = (2 * rmse_avg_spatial * rmse_avg_temporal) / (rmse_avg_spatial + rmse_avg_temporal)

# Print results
print(r2_scores_spatial_all)
print(f'Average r2 score for spatial: {r2_avg_spatial}')
print(f'Average r2 score for temporal: {r2_avg_temporal}')
print(f'f1_score_for_r2: {f1_score_for_r2}')
print(f'Average RMSE score for spatial: {rmse_avg_spatial}')
print(f'Average RMSE score for temporal: {rmse_avg_temporal}')
print(f'f1_score_for_RMSE: {f1_score_for_rmse}')