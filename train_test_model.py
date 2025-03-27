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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('test_data/model_data.csv')
df['home_run_predict'] = (df['home_run'] >= 1).astype(int)
df['game_date'] = pd.to_datetime(df['game_date'])

start_date = df['game_date'].min()
end_date = start_date + pd.DateOffset(years=5)
train_data = df[(df['game_date'] >= start_date) & (df['game_date'] <= end_date)]
features = ['launch_speed_avg', 'launch_angle_avg', 'launch_angle_optimal_avg', 'launch_speed_optimal_avg','AVG', 'OBP', 'SR', 'single_ratio', 'double_ratio', 'triple_ratio', 'home_run_ratio', 'strikeout_ratio', 'out_in_play_ratio', 'field_error_ratio',
'launch_speed_avg_30D', 'launch_angle_avg_30D', 'launch_angle_optimal_avg_30D', 'launch_speed_optimal_avg_30D','AVG_30D', 'OBP_30D', 'SR_30D', 'single_ratio_30D', 'double_ratio_30D', 'triple_ratio_30D', 'home_run_ratio_30D', 'strikeout_ratio_30D', 'out_in_play_ratio_30D', 'field_error_ratio_30D',
'launch_speed_avg_30D_arm', 'launch_angle_avg_30D_arm', 'launch_angle_optimal_avg_30D_arm', 'launch_speed_optimal_avg_30D_arm', 'AVG_30D_arm', 'OBP_30D_arm', 'SR_30D_arm', 'single_ratio_30D_arm', 'double_ratio_30D_arm', 'triple_ratio_30D_arm', 'home_run_ratio_30D_arm', 'strikeout_ratio_30D_arm', 'out_in_play_ratio_30D_arm', 'field_error_ratio_30D_arm',
'launch_speed_avg_hist_arm', 'launch_angle_avg_hist_arm', 'launch_angle_optimal_avg_hist_arm', 'launch_speed_optimal_avg_hist_arm', 'AVG_hist_arm', 'OBP_hist_arm', 'SR_hist_arm', 'single_ratio_hist_arm', 'double_ratio_hist_arm', 'triple_ratio_hist_arm', 'home_run_ratio_hist_arm', 'strikeout_ratio_hist_arm', 'out_in_play_ratio_hist_arm', 'field_error_ratio_hist_arm',
'launch_speed_avg_hist_pitcher', 'launch_angle_avg_hist_pitcher', 'launch_angle_optimal_avg_hist_pitcher', 'launch_speed_optimal_avg_hist_pitcher', 'AVG_hist_pitcher', 'OBP_hist_pitcher', 'SR_hist_pitcher', 'single_ratio_hist_pitcher', 'double_ratio_hist_pitcher', 'triple_ratio_hist_pitcher', 'home_run_ratio_hist_pitcher', 'strikeout_ratio_hist_pitcher', 'out_in_play_ratio_hist_pitcher', 'field_error_ratio_hist_pitcher',
'launch_speed_avg_hist_home', 'launch_angle_avg_hist_home', 'launch_angle_optimal_avg_hist_home', 'launch_speed_optimal_avg_hist_home', 'AVG_hist_home', 'OBP_hist_home', 'SR_hist_home', 'single_ratio_hist_home', 'double_ratio_hist_home', 'triple_ratio_hist_home', 'home_run_ratio_hist_home', 'strikeout_ratio_hist_home', 'out_in_play_ratio_hist_home', 'field_error_ratio_hist_home']
x = train_data[features]
y = train_data['home_run_predict']

model = RandomForestClassifier(random_state=42)
model.fit(x, y)
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
'''
#Decision Tree Model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(x, y)
plt.figure(figsize=(20, 10))  # Set the figure size for better clarity
plot_tree(model, filled=True, feature_names=x.columns, class_names=["No Home Run", "Home Run"], rounded=True, fontsize=6)
plt.show()
'''