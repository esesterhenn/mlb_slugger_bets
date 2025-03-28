import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.inspection import PartialDependenceDisplay
import datetime
import warnings
from joblib import Parallel, delayed


warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('test_data/model_data.csv')
df['home_run_predict'] = (df['home_run'] >= 1).astype(int)
df['game_date'] = pd.to_datetime(df['game_date'])

min_years = 10
min_days = min_years * 365

features = ['launch_speed_avg', 'launch_angle_avg', 'launch_angle_optimal_avg', 'launch_speed_optimal_avg','AVG', 'OBP', 'SR', 'single_ratio', 'double_ratio', 'triple_ratio', 'home_run_ratio', 'strikeout_ratio', 'out_in_play_ratio', 'field_error_ratio',
'launch_speed_avg_30D', 'launch_angle_avg_30D', 'launch_angle_optimal_avg_30D', 'launch_speed_optimal_avg_30D','AVG_30D', 'OBP_30D', 'SR_30D', 'single_ratio_30D', 'double_ratio_30D', 'triple_ratio_30D', 'home_run_ratio_30D', 'strikeout_ratio_30D', 'out_in_play_ratio_30D', 'field_error_ratio_30D',
'launch_speed_avg_30D_arm', 'launch_angle_avg_30D_arm', 'launch_angle_optimal_avg_30D_arm', 'launch_speed_optimal_avg_30D_arm', 'AVG_30D_arm', 'OBP_30D_arm', 'SR_30D_arm', 'single_ratio_30D_arm', 'double_ratio_30D_arm', 'triple_ratio_30D_arm', 'home_run_ratio_30D_arm', 'strikeout_ratio_30D_arm', 'out_in_play_ratio_30D_arm', 'field_error_ratio_30D_arm',
'launch_speed_avg_hist_arm', 'launch_angle_avg_hist_arm', 'launch_angle_optimal_avg_hist_arm', 'launch_speed_optimal_avg_hist_arm', 'AVG_hist_arm', 'OBP_hist_arm', 'SR_hist_arm', 'single_ratio_hist_arm', 'double_ratio_hist_arm', 'triple_ratio_hist_arm', 'home_run_ratio_hist_arm', 'strikeout_ratio_hist_arm', 'out_in_play_ratio_hist_arm', 'field_error_ratio_hist_arm',
'launch_speed_avg_hist_pitcher', 'launch_angle_avg_hist_pitcher', 'launch_angle_optimal_avg_hist_pitcher', 'launch_speed_optimal_avg_hist_pitcher', 'AVG_hist_pitcher', 'OBP_hist_pitcher', 'SR_hist_pitcher', 'single_ratio_hist_pitcher', 'double_ratio_hist_pitcher', 'triple_ratio_hist_pitcher', 'home_run_ratio_hist_pitcher', 'strikeout_ratio_hist_pitcher', 'out_in_play_ratio_hist_pitcher', 'field_error_ratio_hist_pitcher',
'launch_speed_avg_hist_home', 'launch_angle_avg_hist_home', 'launch_angle_optimal_avg_hist_home', 'launch_speed_optimal_avg_hist_home', 'AVG_hist_home', 'OBP_hist_home', 'SR_hist_home', 'single_ratio_hist_home', 'double_ratio_hist_home', 'triple_ratio_hist_home', 'home_run_ratio_hist_home', 'strikeout_ratio_hist_home', 'out_in_play_ratio_hist_home', 'field_error_ratio_hist_home']


# Create training and testing sets
start_date = df['game_date'].min()
end_date = start_date + pd.DateOffset(days=min_days)
filtered_df = df[df['game_date'] > end_date]
unique_sorted_dates = sorted(filtered_df['game_date'].unique())
pred_dates_list = list(unique_sorted_dates)

# Create a function to train and predict
def train_and_predict(date, df, features):
    print(f"Processing date: {date}")
    # Training and testing data for current date
    train_data = df[df['game_date'] < date].copy()
    test_data = df[df['game_date'] == date].copy()
    x_train = train_data[features]
    y_train = train_data['home_run_predict']
    
    # Train model
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, 
                                   min_samples_split=20, min_samples_leaf=10, max_samples=0.8)
    model.fit(x_train, y_train)
    
    # Test model
    x_test = test_data[features]
    test_data['prediction'] = model.predict(x_test)  # Assign predictions directly to test data
    test_data['probability'] = model.predict_proba(x_test)[:, 1]
    
    # Select relevant columns
    test_small = test_data[['game_date', 'batter', 'home_run_predict', 'prediction', 'probability']]
    return test_small

predictions = Parallel(n_jobs=-1)(delayed(train_and_predict)(date, df, features) for date in pred_dates_list)
predictions_df = pd.concat(predictions, ignore_index=True)

# Display final prediction DataFrame
print(predictions_df[predictions_df['batter'] == 668939])
'''
feature_importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the most important features
print("Most Important Features:")
print(feature_importance_df)

# Visualize the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
#plt.show()


PartialDependenceDisplay.from_estimator(model, x, 
                        features=[57],  # Change indices to your most important features
                        feature_names=features,  # Pass your feature names here
                        grid_resolution=50)  # Controls the resolution of the grid
plt.suptitle('Partial Dependence Plots')
plt.show()

#Decision Tree Model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(x, y)
plt.figure(figsize=(20, 10))  # Set the figure size for better clarity
plot_tree(model, filled=True, feature_names=x.columns, class_names=["No Home Run", "Home Run"], rounded=True, fontsize=6)
plt.show()
'''