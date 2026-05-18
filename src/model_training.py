import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing   import StandardScaler

#SETTING UP FOLDER PATHS

BASE_DIR = r"D:\SALES AND DEMAND FORECASTING\SUPER STORE DATA"
BASE_DIR_1 = r"d:\SALES AND DEMAND FORECASTING"
FEATURED_DATA_PATH = os.path.join(BASE_DIR, "featured_data.csv")
MODELS_PATH = os.path.join(BASE_DIR_1, "models")
CHARTS_PATH = os.path.join(BASE_DIR_1, "outputs", "charts")
# Create folders if they don't exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(CHARTS_PATH, exist_ok=True)

#LOADING FEATURED DATA

print("LOADING FEATURED DATA...")
print()

df = pd.read_csv(FEATURED_DATA_PATH)

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
print()

#DEFINING FEATURES AND TARGET

print("DEFINING FEATURES AND TARGET...")
print()

# Features (X) — what the model learns from
FEATURES = [
    'Year',
    'Month',
    'Quarter',
    'Season_Num',
    'Is_Holiday_Season',
    'Is_Peak_Month',
    'Is_Quarter_End',
    'Lag_1',
    'Lag_2',
    'Lag_3',
    'Lag_12',
    'Rolling_3',
    'Rolling_6',
    'Rolling_12',
    'Month_Growth_Rate'
]

# Target (Y) — what we want to predict
TARGET = 'Total_Sales'

X = df[FEATURES]
y = df[TARGET]

print(f"Features selected : {len(FEATURES)}")
print(f"Feature List : {FEATURES}")
print(f"Target column : {TARGET}")
print()

#SPLITTING DATA INTO TRAIN AND TEST

print("SPLITTING DATA INTO TRAIN AND TEST...")
print()

# 80% train, 20% test — NO shuffle for time series data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False       # Important: keep time order intact
)

print(f"Training rows : {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print()

#SCALING THE FEATURES

print("SCALING FEATURES...")
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Features scaled using StandardScaler")
print()

#TRAINING MODEL 1 — LINEAR REGRESSION

print("TRAINING MODEL 1: LINEAR REGRESSION...")
print()

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_predictions = lr_model.predict(X_test_scaled)

# Evaluation
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)
lr_mape = np.mean(np.abs((y_test - lr_predictions) / y_test) * 100)

print(f"Linear Regression Results:")
print(f"MAEMean Absolute Error) : ${lr_mae:,.2f}")
print(f"RMSE(Root Mean Squared Error) : ${lr_rmse:,.2f}")
print(f"R2(Accuracy Score) : {lr_r2:.4f}")
print(f"MAPE (Mean Absolute Percentage Error) : {lr_mape:.2f}%")
print()

#TRAINING MODEL 2 — RANDOM FOREST

print("TRAINING MODEL 2: RANDOM FOREST...")
print()

rf_model = RandomForestRegressor(
    n_estimators=200,     # 200 decision trees
    max_depth=10,         # max depth of each tree
    random_state=42,      # for reproducibility
    n_jobs=1             # use all CPU cores
)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluation
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)
rf_mape = np.mean(np.abs((y_test - rf_predictions) / y_test) * 100)

print(f"Random Forest Results:")
print(f"MAE(Mean Absolute Error) : ${rf_mae:,.2f}")
print(f"RMSE(Root Mean Squared Error) : ${rf_rmse:,.2f}")
print(f"R2(Accuracy Score) : {rf_r2:.4f}")
print(f"MAPE(Mean Absolute Percentage Error) : {rf_mape:.2f}%")
print()

#COMPARING BOTH MODELS

print("MODEL COMPARISON")
print()

comparison = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2 Score', 'MAPE (%)'],
    'Linear Regression': [
        f"${lr_mae:,.2f}",
        f"${lr_rmse:,.2f}",
        f"{lr_r2:.4f}",
        f"{lr_mape:.2f}%"
    ],
    'Random Forest': [
        f"${rf_mae:,.2f}",
        f"${rf_rmse:,.2f}",
        f"{rf_r2:.4f}",
        f"{rf_mape:.2f}%"
    ]
})
print(comparison.to_string(index=False))
print()

# Pick best model
best_model = rf_model if rf_r2 > lr_r2 else lr_model
best_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
best_predictions = rf_predictions if rf_r2 > lr_r2 else lr_predictions
print(f"Best Model: {best_name}")
print()

#SAVING BOTH MODELS

print("SAVING MODELS...")
print()

# Save Linear Regression
lr_path = os.path.join(MODELS_PATH, "linear_regression_model.pkl")
with open(lr_path, 'wb') as f:
    pickle.dump(lr_model, f)
print(f"Linear Regression saved to: {lr_path}")

# Save Random Forest
rf_path = os.path.join(MODELS_PATH, "random_forest_model.pkl")
with open(rf_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Random Forest saved to : {rf_path}")

# Save Scaler
scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to : {scaler_path}")
print()

#CHART 1 — ACTUAL VS PREDICTED

print("GENERATING CHARTS...")
print()

plt.figure(figsize=(14, 6))
plt.plot(y_test.values,
         label='Actual Sales',
         color='steelblue',
         linewidth=2,
         marker='o',
         markersize=5)
plt.plot(best_predictions,
         label=f'Predicted Sales ({best_name})',
         color='orange',
         linewidth=2,
         linestyle='--',
         marker='s',
         markersize=5)
plt.title('Actual vs Predicted Sales', fontsize=16, fontweight='bold')
plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "actual_vs_predicted.png"), dpi=150)
plt.show()
print("Chart 1 saved: actual_vs_predicted.png")

#CHART 2 — FEATURE IMPORTANCE (Random Forest)

feat_importance = pd.Series(rf_model.feature_importances_,index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(10, 7))
feat_importance.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Feature Importance — Random Forest', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "feature_importance.png"), dpi=150)
plt.show()
print("Chart 2 saved: feature_importance.png")

#FORECASTING NEXT 6 MONTHS

print("FORECASTING NEXT 6 MONTHS...")
print()

# Get the last row of data to build future features
last_row    = df.iloc[-1]
last_sales  = df['Total_Sales'].values
last_year   = int(last_row['Year'])
last_month  = int(last_row['Month'])

future_rows = []

for i in range(1, 7):  # forecast 6 months ahead
    # Calculating next month and year
    next_month = (last_month + i - 1) % 12 + 1
    next_year  = last_year + (last_month + i - 1) // 12

    # Recreating features for future month
    quarter          = (next_month - 1) // 3 + 1
    is_holiday       = 1 if next_month in [11, 12] else 0
    is_peak          = 1 if next_month in [3, 4, 9] else 0
    is_quarter_end   = 1 if next_month in [3, 6, 9, 12] else 0
    season_num       = (1 if next_month in [12, 1, 2] else
                        2 if next_month in [3, 4, 5] else
                        3 if next_month in [6, 7, 8] else 4)

    lag1  = last_sales[-1]  if len(last_sales) >= 1  else 0
    lag2  = last_sales[-2]  if len(last_sales) >= 2  else 0
    lag3  = last_sales[-3]  if len(last_sales) >= 3  else 0
    lag12 = last_sales[-12] if len(last_sales) >= 12 else 0
    roll3 = np.mean(last_sales[-3:])  if len(last_sales) >= 3  else lag1
    roll6 = np.mean(last_sales[-6:])  if len(last_sales) >= 6  else lag1
    roll12= np.mean(last_sales[-12:]) if len(last_sales) >= 12 else lag1
    growth= ((lag1 - lag2) / lag2 * 100) if lag2 != 0 else 0

    future_rows.append({
        'Year' : next_year,
        'Month' : next_month,
        'Quarter': quarter,
        'Season_Num': season_num,
        'Is_Holiday_Season': is_holiday,
        'Is_Peak_Month': is_peak,
        'Is_Quarter_End': is_quarter_end,
        'Lag_1': lag1,
        'Lag_2': lag2,
        'Lag_3': lag3,
        'Lag_12': lag12,
        'Rolling_3' : roll3,
        'Rolling_6': roll6,
        'Rolling_12': roll12,
        'Month_Growth_Rate': growth
    })

    # Adding predicted sales back so next month uses it as lag
    future_pred = rf_model.predict(pd.DataFrame([future_rows[-1]]))[0]
    last_sales  = np.append(last_sales, future_pred)

future_df = pd.DataFrame(future_rows)
future_df['Forecast_Sales'] = rf_model.predict(future_df[FEATURES]).round(2)
future_df['Month_Label'] = (future_df['Year'].astype(str) + '-' + future_df['Month'].astype(str).str.zfill(2))
print(future_df[['Month_Label', 'Forecast_Sales']].to_string(index=False))
print()

#CHART 3 — FUTURE FORECAST
#Historical monthly sales
hist = df[['Year', 'Month', 'Total_Sales']].copy()
hist['Month_Label'] = (hist['Year'].astype(str) + '-' + hist['Month'].astype(str).str.zfill(2))

plt.figure(figsize=(16, 6))
plt.plot(hist['Month_Label'],hist['Total_Sales'],
         label='Historical Sales',
         color='steelblue',
         linewidth=2)
plt.plot(future_df['Month_Label'], future_df['Forecast_Sales'],
         label='Forecasted Sales (Next 6 Months)',
         color='orange',
         linewidth=2,
         linestyle='--',
         marker='o',
         markersize=7)
plt.axvline(x=hist['Month_Label'].iloc[-1],
            color='red',
            linestyle=':',
            linewidth=1.5,
            label='Forecast Start')
plt.title('Sales Forecast — Next 6 Months', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "forecast_future.png"), dpi=150)
plt.show()
print("Chart 3 saved: forecast_future.png")

#FINAL SUMMARY

print()
print("MODEL TRAINING COMPLETE — FINAL SUMMARY")
print()
print(f"Best Model: {best_name}")
print(f"R2 Score : {max(lr_r2, rf_r2):.4f}")
print(f"MAE : ${min(lr_mae, rf_mae):,.2f}")
print(f"MAPE : {min(lr_mape, rf_mape):.2f}%")
print()
print("6 Month Forecast:")
for _, row in future_df.iterrows():
    print(f"{row['Month_Label']} - ${row['Forecast_Sales']:,.2f}")