import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#SETTING UP PATHS

BASE_DIR = r"D:\SALES AND DEMAND FORECASTING\SUPER STORE DATA"
BASE_DIR_1 = r"D:\SALES AND DEMAND FORECASTING"
FEATURED_DATA_PATH = os.path.join(BASE_DIR, "featured_data.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
MODELS_PATH = os.path.join(BASE_DIR_1, "models")
CHARTS_PATH = os.path.join(BASE_DIR_1, "outputs", "charts")
REPORTS_PATH = os.path.join(BASE_DIR_1, "outputs", "reports")

os.makedirs(CHARTS_PATH,  exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']

#LOADING DATA AND MODELS

print("LOADING DATA AND MODELS...")
print()

# Loading featured data
df = pd.read_csv(FEATURED_DATA_PATH)

# Loading cleaned data for category/region breakdowns
raw = pd.read_csv(CLEANED_DATA_PATH, encoding='latin-1')
raw['Order Date'] = pd.to_datetime(raw['Order Date'])
raw['Year'] = raw['Order Date'].dt.year
raw['Month'] = raw['Order Date'].dt.month

# Loading Random Forest model
rf_path = os.path.join(MODELS_PATH, "random_forest_model.pkl")
with open(rf_path, 'rb') as f:
    rf_model = pickle.load(f)

# Loading scaler
scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print(f"Featured data loaded : {df.shape[0]} rows")
print(f"Cleaned data loaded : {raw.shape[0]} rows")
print(f"Random Forest loaded")
print()

#DEFINING FEATURES

FEATURES = [
    'Year', 'Month', 'Quarter', 'Season_Num',
    'Is_Holiday_Season', 'Is_Peak_Month', 'Is_Quarter_End',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
    'Rolling_3', 'Rolling_6', 'Rolling_12',
    'Month_Growth_Rate'
]

X = df[FEATURES]
y = df['Total_Sales']
predictions = rf_model.predict(X)

#CHART 1 — FULL SALES TREND OVERVIEW

print("GENERATING CHARTS...")
print()

df['Month_Label'] = (df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2))

plt.figure(figsize=(16, 6))
plt.plot(df['Month_Label'], df['Total_Sales'],
         color=COLORS[0], linewidth=2.5,
         marker='o', markersize=4,
         label='Actual Sales')
plt.plot(df['Month_Label'], predictions,
         color=COLORS[1], linewidth=2,
         linestyle='--', marker='s', markersize=4,
         label='Model Prediction')
plt.fill_between(df['Month_Label'],
                 df['Total_Sales'], predictions,
                 alpha=0.1, color='red',
                 label='Prediction Error')
plt.title('Sales Trend: Actual vs Model Predictions',
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH,
            "sales_trend_overview.png"), dpi=150)
plt.show()
print("Chart 1 saved: sales_trend_overview.png")

#Chart 2: sales by region
region_sales = (
    raw.groupby('Region')['Sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

fig.suptitle('Sales by Region', fontsize=16,
             fontweight='bold', y=0.95)

plt.subplots_adjust(top=0.85)

# Bar chart
axes[0].bar(region_sales['Region'],
            region_sales['Sales'],
            color=COLORS[:len(region_sales)],
            edgecolor='white', linewidth=0.8)
axes[0].set_title('Total Sales by Region', fontsize=13)
axes[0].set_xlabel('Region', fontsize=11)
axes[0].set_ylabel('Total Sales ($)', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(region_sales['Sales']):
    axes[0].text(i, v + 1000, f'${v:,.0f}',
                 ha='center', fontsize=9, fontweight='bold')

# Pie chart
axes[1].pie(region_sales['Sales'],
            labels=region_sales['Region'],
            colors=COLORS[:len(region_sales)],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Sales Distribution by Region', fontsize=13)

plt.savefig(os.path.join(CHARTS_PATH, "sales_by_region.png"),
            dpi=150, bbox_inches='tight')
plt.show()
print("Chart 2 saved: sales_by_region.png")

#CHART 3 — SALES BY CATEGORY

category_sales = (
    raw.groupby('Category')['Sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Sales by Category', fontsize=16,
             fontweight='bold')

# Bar chart
axes[0].barh(category_sales['Category'],
             category_sales['Sales'],
             color=COLORS[:len(category_sales)],
             edgecolor='white')
axes[0].set_title('Total Sales by Category', fontsize=13)
axes[0].set_xlabel('Total Sales ($)', fontsize=11)
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(category_sales['Sales']):
    axes[0].text(v + 500, i, f'${v:,.0f}',
                 va='center', fontsize=9, fontweight='bold')

# Pie chart
axes[1].pie(category_sales['Sales'],
            labels=category_sales['Category'],
            colors=COLORS[:len(category_sales)],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Sales Share by Category', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH,
            "sales_by_category.png"), dpi=150)
plt.show()
print("Chart 3 saved: sales_by_category.png")

#CHART 4 — YEARLY GROWTH

yearly_sales = (
    raw.groupby('Year')['Sales']
    .sum()
    .reset_index()
)
yearly_sales['Growth_%'] = (
    yearly_sales['Sales'].pct_change() * 100
).round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Yearly Sales Growth', fontsize=16,
             fontweight='bold')

# Sales bars
bars = axes[0].bar(yearly_sales['Year'].astype(str),
                   yearly_sales['Sales'],
                   color=COLORS[:len(yearly_sales)],
                   edgecolor='white', width=0.5)
axes[0].set_title('Total Sales per Year', fontsize=13)
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Total Sales ($)', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)
for bar, val in zip(bars, yearly_sales['Sales']):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1000,
                 f'${val:,.0f}',
                 ha='center', fontsize=9, fontweight='bold')

# Growth % line
growth_data = yearly_sales.dropna(subset=['Growth_%'])
axes[1].plot(growth_data['Year'].astype(str),
             growth_data['Growth_%'],
             color=COLORS[1], linewidth=2.5,
             marker='o', markersize=8)
axes[1].axhline(y=0, color='red',
                linestyle='--', linewidth=1)
axes[1].set_title('Year-over-Year Growth (%)', fontsize=13)
axes[1].set_xlabel('Year', fontsize=11)
axes[1].set_ylabel('Growth (%)', fontsize=11)
axes[1].grid(alpha=0.3)
for x, y_val in zip(growth_data['Year'].astype(str),
                    growth_data['Growth_%']):
    axes[1].annotate(f'{y_val:.1f}%',
                     xy=(x, y_val),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center', fontsize=10,
                     fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH,
            "yearly_growth.png"), dpi=150)
plt.show()
print("Chart 4 saved: yearly_growth.png")

#CHART 5 — FORECAST DASHBOARD

print("GENERATING 6-MONTH FORECAST...")
print()

# Get last row values from featured data
last_sales  = df['Total_Sales'].values
last_year   = int(df.iloc[-1]['Year'])
last_month  = int(df.iloc[-1]['Month'])

future_rows = []

for i in range(1, 7):  # 6 months ahead
    # Calculate next month and year
    next_month = (last_month + i - 1) % 12 + 1
    next_year  = last_year + (last_month + i - 1) // 12

    # Recreate all features for future month
    quarter        = (next_month - 1) // 3 + 1
    is_holiday     = 1 if next_month in [11, 12] else 0
    is_peak        = 1 if next_month in [3, 4, 9]  else 0
    is_qtr_end     = 1 if next_month in [3, 6, 9, 12] else 0
    season_num     = (1 if next_month in [12, 1, 2] else
                      2 if next_month in [3, 4, 5]  else
                      3 if next_month in [6, 7, 8]  else 4)

    lag1   = last_sales[-1]
    lag2   = last_sales[-2]  if len(last_sales) >= 2  else lag1
    lag3   = last_sales[-3]  if len(last_sales) >= 3  else lag1
    lag12  = last_sales[-12] if len(last_sales) >= 12 else lag1
    roll3  = np.mean(last_sales[-3:])
    roll6  = np.mean(last_sales[-6:])
    roll12 = np.mean(last_sales[-12:])
    growth = ((lag1 - lag2) / lag2 * 100) if lag2 != 0 else 0

    row = {
        'Year'             : next_year,
        'Month'            : next_month,
        'Quarter'          : quarter,
        'Season_Num'       : season_num,
        'Is_Holiday_Season': is_holiday,
        'Is_Peak_Month'    : is_peak,
        'Is_Quarter_End'   : is_qtr_end,
        'Lag_1'            : lag1,
        'Lag_2'            : lag2,
        'Lag_3'            : lag3,
        'Lag_12'           : lag12,
        'Rolling_3'        : roll3,
        'Rolling_6'        : roll6,
        'Rolling_12'       : roll12,
        'Month_Growth_Rate': growth
    }
    future_rows.append(row)

    # Add predicted value back for next iteration
    pred       = rf_model.predict(pd.DataFrame([row]))[0]
    last_sales = np.append(last_sales, pred)

future_df = pd.DataFrame(future_rows)
future_df['Forecast_Sales'] = rf_model.predict(
    future_df[FEATURES]).round(2)
future_df['Month_Label'] = (
    future_df['Year'].astype(str) + '-' +
    future_df['Month'].astype(str).str.zfill(2)
)

# Build dashboard
fig = plt.figure(figsize=(18, 14))

fig.suptitle('Sales Forecasting Dashboard',
             fontsize=20, fontweight='bold', y=0.95)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.45, wspace=0.3)

plt.subplots_adjust(top=0.88)

# Panel 1: Historical + Forecast
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['Month_Label'], df['Total_Sales'],
         color=COLORS[0], linewidth=2,
         label='Historical Sales', marker='o', markersize=3)
ax1.plot(future_df['Month_Label'],
         future_df['Forecast_Sales'],
         color=COLORS[1], linewidth=2.5,
         linestyle='--', marker='o', markersize=7,
         label='6-Month Forecast')
ax1.axvline(x=df['Month_Label'].iloc[-1],
            color='red', linestyle=':', linewidth=2,
            label='Forecast Start')
ax1.fill_between(future_df['Month_Label'],
                 future_df['Forecast_Sales'] * 0.9,
                 future_df['Forecast_Sales'] * 1.1,
                 alpha=0.2, color=COLORS[1],
                 label='±10% Confidence Band')
ax1.set_title('Historical Sales + 6-Month Forecast',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Sales ($)')
ax1.legend(fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# Panel 2: Forecast bar chart
ax2 = fig.add_subplot(gs[1, 0])
bars = ax2.bar(future_df['Month_Label'],
               future_df['Forecast_Sales'],
               color=COLORS[1], edgecolor='white',
               alpha=0.85)
ax2.set_title('Forecasted Sales — Next 6 Months',
              fontsize=13, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Forecasted Sales ($)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, future_df['Forecast_Sales']):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 200,
             f'${val:,.0f}',
             ha='center', fontsize=8, fontweight='bold')

# Panel 3: Model metrics
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
mae  = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2   = r2_score(y, predictions)
mape = np.mean(np.abs((y - predictions) / y) * 100)
total_forecast = future_df['Forecast_Sales'].sum()
avg_forecast   = future_df['Forecast_Sales'].mean()
best_month     = future_df.loc[
    future_df['Forecast_Sales'].idxmax(), 'Month_Label']

metrics_text = f"""
  MODEL PERFORMANCE
  ─────────────────────────────
  R² Score         :  {r2:.4f}
  MAE              :  ${mae:,.2f}
  RMSE             :  ${rmse:,.2f}
  MAPE             :  {mape:.2f}%

  6-MONTH FORECAST SUMMARY
  ─────────────────────────────
  Total Forecast   :  ${total_forecast:,.2f}
  Monthly Average  :  ${avg_forecast:,.2f}
  Best Month       :  {best_month}
  ─────────────────────────────
"""
ax3.text(0.05, 0.95, metrics_text,
         transform=ax3.transAxes,
         fontsize=11, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round',
                   facecolor='#f0f4f8',
                   alpha=0.8))
ax3.set_title('Model Summary', fontsize=13,
              fontweight='bold')

plt.savefig(os.path.join(CHARTS_PATH,
            "forecast_dashboard.png"),
            dpi=150, bbox_inches='tight')
plt.show()
print("Chart 5 saved: forecast_dashboard.png")

#GENERATING BUSINESS REPORT

print("GENERATING BUSINESS REPORT...")
print()

# Calculating key business numbers
total_revenue    = raw['Sales'].sum()
avg_order_value  = raw['Sales'].mean()
best_region      = region_sales.iloc[0]['Region']
best_category    = category_sales.iloc[0]['Category']
best_year        = yearly_sales.loc[
    yearly_sales['Sales'].idxmax(), 'Year']
best_year_sales  = yearly_sales['Sales'].max()
last_year_sales  = yearly_sales.iloc[-1]['Sales']
prev_year_sales  = yearly_sales.iloc[-2]['Sales']
yoy_growth       = ((last_year_sales - prev_year_sales)
                    / prev_year_sales * 100)

report = f"""
╔══════════════════════════════════════════════════════════╗
║           SALES FORECASTING — BUSINESS REPORT           ║
╚══════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BUSINESS OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total Revenue (All Years)  :  ${total_revenue:,.2f}
  Average Order Value        :  ${avg_order_value:,.2f}
  Best Performing Region     :  {best_region}
  Best Performing Category   :  {best_category}
  Best Sales Year            :  {best_year} (${best_year_sales:,.2f})
  Year-over-Year Growth      :  {yoy_growth:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Model Used    :  Random Forest Regressor
  R² Score      :  {r2:.4f}  (closer to 1.0 = better)
  MAE           :  ${mae:,.2f} (average prediction error)
  RMSE          :  ${rmse:,.2f} (penalizes large errors)
  MAPE          :  {mape:.2f}% (error as % of actual sales)

  Interpretation:
  The model explains {r2*100:.1f}% of sales variation.
  On average, predictions are off by ${mae:,.2f} per month.
  A MAPE of {mape:.2f}% means predictions are
  {"excellent (under 10%)" if mape < 10 else
   "good (under 20%)"      if mape < 20 else
   "acceptable (under 30%)" if mape < 30 else
   "needs improvement"}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. 6-MONTH SALES FORECAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
for _, row in future_df.iterrows():
    report += (f"  {row['Month_Label']}  →  "
               f"${row['Forecast_Sales']:>12,.2f}\n")

report += f"""
  ─────────────────────────────────────────
  Total 6-Month Forecast  :  ${total_forecast:,.2f}
  Monthly Average         :  ${avg_forecast:,.2f}
  Best Forecast Month     :  {best_month}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. BUSINESS RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INVENTORY PLANNING
  → Stock up 4-6 weeks before peak forecast months
  → Reduce orders during low forecast months
  → Focus inventory on {best_category} (top category)

  CASH FLOW PLANNING
  → Expected revenue next 6 months: ${total_forecast:,.2f}
  → Plan expenses around avg monthly income: ${avg_forecast:,.2f}
  → Strongest month ahead: {best_month}

  STAFFING
  → Increase staff in high-forecast months
  → Plan training and onboarding in low months
  → Focus sales team on {best_region} region (top performer)

  MARKETING
  → Run promotions before low-sales forecast months
  → Increase ad spend ahead of peak months
  → Target high-value customers in {best_region}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  → Sales grow consistently year over year ({yoy_growth:.1f}% last year)
  → Holiday season (Nov-Dec) drives highest monthly sales
  → {best_region} region contributes the most revenue
  → {best_category} is the strongest product category
  → Model predicts ${total_forecast:,.2f} over next 6 months

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Report generated by: Sales Forecasting ML System
  Model: Random Forest Regressor
╚══════════════════════════════════════════════════════════╝
"""

# Save report
report_path = os.path.join(REPORTS_PATH, "business_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"Business report saved to:")
print(f"{report_path}")

# FINAL SUMMARY

print("VISUALIZATION COMPLETE — ALL FILES SAVED")
print()
print(f"Charts saved to  : {CHARTS_PATH}")
print(f"Report saved to  : {REPORTS_PATH}")
print()
print("Charts Generated:")
print("sales_trend_overview.png")
print("sales_by_region.png")
print("sales_by_category.png")
print("yearly_growth.png")
print("forecast_dashboard.png")
print()
print("Report Generated:")
