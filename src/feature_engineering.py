import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#SETTING UP FOLDER PATHS

BASE_DIR = r"d:\SALES AND DEMAND FORECASTING\SUPER STORE DATA"
BASE_DIR_1 = r"d:\SALES AND DEMAND FORECASTING"
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
FEATURED_DATA_PATH = os.path.join(BASE_DIR, "featured_data.csv")
MONTHLY_DATA_PATH = os.path.join(BASE_DIR, "monthly_sales.csv")
CHARTS_PATH = os.path.join(BASE_DIR_1, "outputs", "charts")

#LOADING CLEANED DATA

print("LOADING CLEANED DATA...")

df = pd.read_csv(CLEANED_DATA_PATH, encoding='latin-1')

# Re-convert Order Date back to datetime (CSV saves it as string)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print()

#BASIC DATE FEATURES

print("CREATING BASIC DATE FEATURES...")

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Quarter'] = df['Order Date'].dt.quarter
df['Day_of_Week'] = df['Order Date'].dt.dayofweek
df['Week_Number'] = df['Order Date'].dt.isocalendar().week.astype(int)
df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
df['Days_to_Ship'] = (df['Ship Date'] - df['Order Date']).dt.days

print("Created: Year, Month, Day, Quarter")
print("Created: Day_of_Week( 0=Monday, 6=Sunday), Week_Number")
print("Created: Is_Weekend, Days_to_Ship")
print()

#SEASON FEATURE

print("CREATING SEASON FEATURE...")

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)

# Convert Season to a number so the model can use it
season_map = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
df['Season_Num'] = df['Season'].map(season_map)

print("Created: Season (Winter/Spring/Summer/Fall)")
print("Created: Season_Num (1/2/3/4)")
print(df['Season'].value_counts())
print()

#HOLIDAY / PEAK SEASON FLAGS

print("CREATING HOLIDAY & PEAK SEASON FLAGS...")
print()

# November & December = Holiday season (Black Friday, Christmas)
df['Is_Holiday_Season'] = df['Month'].apply(lambda x: 1 if x in [11, 12] else 0)

# March, April, September = Back to school / new quarter
df['Is_Peak_Month'] = df['Month'].apply(lambda x: 1 if x in [3, 4, 9] else 0)

# End of quarter months
df['Is_Quarter_End'] = df['Month'].apply(lambda x: 1 if x in [3, 6, 9, 12] else 0)

print("Created: Is_Holiday_Season (Nov & Dec = 1)")
print("Created: Is_Peak_Month (Mar, Apr, Sep = 1)")
print("Created: Is_Quarter_End (Mar, Jun, Sep, Dec = 1)")
print()

#AGGREGATE INTO MONTHLY SALES

print("BUILDING MONTHLY SALES TABLE...")
print()

monthly = (
    df.groupby(['Year', 'Month', 'Quarter', 'Season','Season_Num', 'Is_Holiday_Season','Is_Peak_Month', 'Is_Quarter_End'])['Sales'].sum().reset_index())
monthly.columns = [
    'Year', 'Month', 'Quarter', 'Season',
    'Season_Num', 'Is_Holiday_Season',
    'Is_Peak_Month', 'Is_Quarter_End', 'Total_Sales'
]
monthly['Total_Sales'] = monthly['Total_Sales'].round(2)
monthly = monthly.sort_values(['Year', 'Month']).reset_index(drop=True)

print(monthly)
print()

#LAG FEATURES (past sales as features)

print("CREATING LAG FEATURES...")
print()

# Lag 1: Sales from 1 month ago
monthly['Lag_1'] = monthly['Total_Sales'].shift(1)

# Lag_2: Sales from 2 months ago
monthly['Lag_2'] = monthly['Total_Sales'].shift(2)

# Lag 3: Sales from 3 months ago
monthly['Lag_3'] = monthly['Total_Sales'].shift(3)

# Lag 12: Sales from same month last year
monthly['Lag_12'] = monthly['Total_Sales'].shift(12)

print("Created: Lag_1  (sales 1 month ago)")
print("Created: Lag_2  (sales 2 months ago)")
print("Created: Lag_3  (sales 3 months ago)")
print("Created: Lag_12 (sales same month last year)")
print()

#ROLLING AVERAGE FEATURES (smoothed trends)

print("CREATING ROLLING AVERAGE FEATURES...")
print()

# 3-month rolling average
monthly['Rolling_3']  = monthly['Total_Sales'].shift(1).rolling(window=3).mean().round(2)

# 6-month rolling average
monthly['Rolling_6']  = monthly['Total_Sales'].shift(1).rolling(window=6).mean().round(2)

# 12-month rolling average (annual trend)
monthly['Rolling_12'] = monthly['Total_Sales'].shift(1).rolling(window=12).mean().round(2)

print("Created: Rolling_3  (3-month average trend)")
print("Created: Rolling_6  (6-month average trend)")
print("Created: Rolling_12 (12-month average trend)")
print()

#GROWTH RATE FEATURE

print("CREATING GROWTH RATE FEATURE...")
print()

monthly['Month_Growth_Rate'] = ((monthly['Total_Sales'] - monthly['Lag_1']) / monthly['Lag_1'] * 100).round(2)

print("Created: Month_Growth_Rate (percentage change from last month)")
print()

#DROP ROWS WITH NaN FROM LAG FEATURES

print("CLEANING UP NaN VALUES...")
print()

before = len(monthly)
monthly = monthly.dropna()
after  = len(monthly)

print(f"Removed {before - after} rows with NaN (from lag features)")
print(f"Remaining rows: {after}")
print()

#FINAL FEATURE SUMMARY

print("FINAL FEATURE LIST")
print()
for i, col in enumerate(monthly.columns, 1):
    print(f"   {i:02d}. {col}")
print()

#SAVING FEATURED DATA

print("SAVING FEATURED DATA...")
print()

monthly.to_csv(FEATURED_DATA_PATH, index=False)
print(f"Featured data saved to:")
print(f"{FEATURED_DATA_PATH}")
print()

#VISUALIZATIONS

print("GENERATING CHARTS...")
print()

#CHART 1: Sales by Month (Average across all years)
plt.figure(figsize=(12, 5))
monthly_avg = monthly.groupby('Month')['Total_Sales'].mean()
plt.bar(monthly_avg.index, monthly_avg.values, color='steelblue', edgecolor='white')
plt.title('Average Sales by Month', fontsize=15, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Sales ($)', fontsize=12)
plt.xticks(range(1, 13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "sales_by_month.png"), dpi=150)
plt.show()
print("Chart 1 saved: sales_by_month.png")

#CHART 2: Sales by Quarter 
plt.figure(figsize=(8, 5))
quarterly_avg = monthly.groupby('Quarter')['Total_Sales'].mean()
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], quarterly_avg.values, color=colors, edgecolor='white')
plt.title('Average Sales by Quarter', fontsize=15, fontweight='bold')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Average Sales ($)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "sales_by_quarter.png"), dpi=150)
plt.show()
print("Chart 2 saved: sales_by_quarter.png")

#CHART 3: Sales by Season 
plt.figure(figsize=(8, 5))
season_avg = monthly.groupby('Season')['Total_Sales'].mean()
colors2 = ['#e15759', '#76b7b2', '#f28e2b', '#4e79a7']
plt.bar(season_avg.index, season_avg.values, color=colors2, edgecolor='white')
plt.title('Average Sales by Season', fontsize=15, fontweight='bold')
plt.xlabel('Season', fontsize=12)
plt.ylabel('Average Sales ($)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "sales_by_season.png"), dpi=150)
plt.show()
print("Chart 3 saved: sales_by_season.png")

#CHART 4: Correlation Heatmap 
plt.figure(figsize=(12, 8))
numeric_cols = monthly.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',linewidths=0.5, square=True)
plt.title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "correlation_heatmap.png"), dpi=150)
plt.show()
print("Chart 4 saved: correlation_heatmap.png")
print()

# FINAL SUMMARY

print("FEATURE ENGINEERING COMPLETE — SUMMARY")
print()
print(f"Total Monthly Records : {len(monthly)}")
print(f"Total Features Created: {len(monthly.columns)}")
print(f"Year Range            : {int(monthly['Year'].min())} - {int(monthly['Year'].max())}")
print(f"Highest Sales Month   : ${monthly['Total_Sales'].max():,.2f}")
print(f"Lowest Sales Month    : ${monthly['Total_Sales'].min():,.2f}")
print(f"Average Monthly Sales : ${monthly['Total_Sales'].mean():,.2f}")
