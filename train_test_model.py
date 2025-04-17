import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.inspection import PartialDependenceDisplay
import datetime
import warnings
from joblib import Parallel, delayed


'''
FEATURE IDEAS
Pitcher arsenal percent (what percent of pitches are each pitch type?)
Batter vs pitch type
'''


warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('model_data.csv')
df['home_run_predict'] = (df['home_run'] >= 1).astype(int)
df['game_date'] = pd.to_datetime(df['game_date'])

min_years = 5

overall_7d = ['launch_speed_avg', 'launch_angle_avg', 'launch_angle_optimal_avg', 'launch_speed_optimal_avg','AVG', 'OBP', 'SR', 'single_ratio', 'double_ratio', 'triple_ratio', 'home_run_ratio', 'strikeout_ratio', 'out_in_play_ratio', 'field_error_ratio']
overall_30d = ['launch_speed_avg_30D', 'launch_angle_avg_30D', 'launch_angle_optimal_avg_30D', 'launch_speed_optimal_avg_30D','AVG_30D', 'OBP_30D', 'SR_30D', 'single_ratio_30D', 'double_ratio_30D', 'triple_ratio_30D', 'home_run_ratio_30D', 'strikeout_ratio_30D', 'out_in_play_ratio_30D', 'field_error_ratio_30D']
pitch_arm_30d = ['launch_speed_avg_30D_arm', 'launch_angle_avg_30D_arm', 'launch_angle_optimal_avg_30D_arm', 'launch_speed_optimal_avg_30D_arm', 'AVG_30D_arm', 'OBP_30D_arm', 'SR_30D_arm', 'single_ratio_30D_arm', 'double_ratio_30D_arm', 'triple_ratio_30D_arm', 'home_run_ratio_30D_arm', 'strikeout_ratio_30D_arm', 'out_in_play_ratio_30D_arm', 'field_error_ratio_30D_arm']
pitch_arm_all = ['launch_speed_avg_hist_arm', 'launch_angle_avg_hist_arm', 'launch_angle_optimal_avg_hist_arm', 'launch_speed_optimal_avg_hist_arm', 'AVG_hist_arm', 'OBP_hist_arm', 'SR_hist_arm', 'single_ratio_hist_arm', 'double_ratio_hist_arm', 'triple_ratio_hist_arm', 'home_run_ratio_hist_arm', 'strikeout_ratio_hist_arm', 'out_in_play_ratio_hist_arm', 'field_error_ratio_hist_arm']
pitcher_all = ['launch_speed_avg_hist_pitcher', 'launch_angle_avg_hist_pitcher', 'launch_angle_optimal_avg_hist_pitcher', 'launch_speed_optimal_avg_hist_pitcher', 'AVG_hist_pitcher', 'OBP_hist_pitcher', 'SR_hist_pitcher', 'single_ratio_hist_pitcher', 'double_ratio_hist_pitcher', 'triple_ratio_hist_pitcher', 'home_run_ratio_hist_pitcher', 'strikeout_ratio_hist_pitcher', 'out_in_play_ratio_hist_pitcher', 'field_error_ratio_hist_pitcher']
home_team_all = ['launch_speed_avg_hist_home', 'launch_angle_avg_hist_home', 'launch_angle_optimal_avg_hist_home', 'launch_speed_optimal_avg_hist_home', 'AVG_hist_home', 'OBP_hist_home', 'SR_hist_home', 'single_ratio_hist_home', 'double_ratio_hist_home', 'triple_ratio_hist_home', 'home_run_ratio_hist_home', 'strikeout_ratio_hist_home', 'out_in_play_ratio_hist_home', 'field_error_ratio_hist_home']

# Create training and testing sets
start_date = df['game_date'].min()
print(start_date)
end_date = df['game_date'].max()
print(end_date)
start_year = start_date.year
end_year = start_year + min_years
year_list = list(range(end_year,end_date.year+1))
print(year_list)
df['year'] = df['game_date'].dt.year

# Create a function to train and predict
def train_and_predict(year, df, features):
    print(f"Processing date: {year}")
    # Training and testing data for current date
    df = df.dropna(subset=features)
    train_data = df[df['year'] < year].copy()
    unique_dates = sorted(train_data['year'].unique())
    date_to_index = {date: idx for idx, date in enumerate(unique_dates)}
    train_data['recency_index'] = train_data['year'].map(date_to_index)
    train_data['model_weight'] = (train_data['recency_index']) / (train_data['recency_index']).sum()

    test_data = df[df['year'] == year].copy()
    x_train = train_data[features]
    y_train = train_data['home_run_predict']
    x_cal = x_train.copy()
    y_cal = y_train.copy()

    # Train model
    model = RandomForestClassifier(random_state=42, n_estimators=200, 
                                   min_samples_split=30, min_samples_leaf=8, max_samples=0.8)
    model.fit(x_train, y_train, sample_weight=train_data['model_weight'])
    calibrated_rf = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    calibrated_rf.fit(x_cal, y_cal)

    # Test model
    x_test = test_data[features]
    #test_data['prediction'] = model.predict(x_test)  # Assign predictions directly to test data
    #test_data['probability'] = model.predict_proba(x_test)[:, 1]
    test_data['probability'] = calibrated_rf.predict_proba(x_test)[:, 1]

    # Select relevant columns
    test_small = test_data[['game_date', 'batter','batter_first_name','batter_last_name','game_num_in_day','game_pk','home_team', 'home_run_predict', 'probability']]
    return test_small

pk_df = df[df['year'] >= end_year]
pk_df = pk_df[['game_date', 'batter','batter_first_name','batter_last_name','game_num_in_day','game_pk','home_team', 'home_run_predict']]

# Define features with all included
features = overall_7d + overall_30d + pitch_arm_30d + pitch_arm_all + pitcher_all + home_team_all
predictions = Parallel(n_jobs=-1)(delayed(train_and_predict)(year, df, features) for year in year_list)
predictions_df_all = pd.concat(predictions, ignore_index=True)

# Define features with all included except pitcher vs. batter data
features = overall_7d + overall_30d + pitch_arm_30d + pitch_arm_all + home_team_all
predictions = Parallel(n_jobs=-1)(delayed(train_and_predict)(year, df, features) for year in year_list)
predictions_df_no_pitch = pd.concat(predictions, ignore_index=True)


predictions_df = pd.merge(pk_df,predictions_df_all[['game_date','batter','game_num_in_day','game_pk','probability']],on=['game_date','batter','game_num_in_day','game_pk'], how="left", suffixes=("","_1"))
predictions_df = pd.merge(predictions_df,predictions_df_no_pitch[['game_date','batter','game_num_in_day','game_pk','probability']],on=['game_date','batter','game_num_in_day','game_pk'], how="left", suffixes=("","_2"))
predictions_df = predictions_df.rename(columns={
    'probability': 'probability_1'})

predictions_df['probability'] = predictions_df[['probability_1', 'probability_2']].bfill(axis=1).iloc[:, 0]

predictions_df = predictions_df.dropna(subset=['probability'])

print(predictions_df.head(10))

odds_df = pd.read_csv('formatted_odds.csv')
odds_df['DraftKings.over_prob'] = 1/odds_df['DraftKings.over_price']
odds_df['DraftKings.under_prob'] = 1/odds_df['DraftKings.under_price']
odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

predictions_df = predictions_df.merge(odds_df[['game_date','batter','game_num_in_day','game_pk','home_team','DraftKings.point','DraftKings.over_price','DraftKings.under_price','DraftKings.over_prob','DraftKings.under_prob']], on=['game_date','batter','game_num_in_day','game_pk','home_team'], how="left")

print(predictions_df[predictions_df['game_date']=='2023-05-03'])

predictions_df['std_prediction'] = (predictions_df['probability']>0.4).astype(int)
conditions = [
    predictions_df['probability'] > predictions_df['DraftKings.over_prob'],
    (1-predictions_df['probability']) > predictions_df['DraftKings.under_prob']
]
predictions_df['odds_prediction'] = np.select(conditions, [1,0], default=np.nan)

# Display final prediction DataFrame

decision_matrix = pd.crosstab(predictions_df['home_run_predict'], predictions_df['odds_prediction'])
accuracy = np.mean(predictions_df['home_run_predict'] == predictions_df['odds_prediction'])

hr_bet = predictions_df[(predictions_df['odds_prediction'] == 1) & (predictions_df['DraftKings.over_price'].notnull())]
hr_bet['odds_difference'] = hr_bet['probability'] - hr_bet['DraftKings.over_prob']
hr_bet['bet_amount_optimal'] = hr_bet.groupby('game_date')['odds_difference'].transform(
    lambda x: (x / x.sum()) * len(x)
)
hr_bet['correct_prediction_naive'] = hr_bet['home_run_predict'] * hr_bet['odds_prediction']
hr_bet['winnings_naive'] = hr_bet['correct_prediction_naive'] * (hr_bet['DraftKings.over_price'])
hr_bet['correct_prediction_optimal'] = hr_bet['home_run_predict'] * hr_bet['bet_amount_optimal']
hr_bet['winnings_optimal'] = hr_bet['correct_prediction_optimal'] * (hr_bet['DraftKings.over_price'])

average_winnings = hr_bet['winnings_naive'].sum() / hr_bet['DraftKings.over_price'].notnull().sum()
print('Naive Profitability:' + str(average_winnings))

optimal_winnings = hr_bet['winnings_optimal'].sum() / hr_bet['DraftKings.over_price'].notnull().sum()
print('Optimal Profitability:' + str(optimal_winnings))

mean_prob = predictions_df['probability'].mean()
predictions_df['probability_accuracy'] = (predictions_df['probability'] - mean_prob) * predictions_df['home_run_predict']
mean_prob_acc = predictions_df['probability_accuracy'].mean()
print(decision_matrix)
print(accuracy)
print(mean_prob_acc)

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