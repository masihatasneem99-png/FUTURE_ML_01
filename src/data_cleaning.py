import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Setting folder paths

# Base project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: raw data
RAW_DATA_PATH = os.path.join(BASE_DIR, "SUPER STORE DATA", "Sample - Superstore.csv")
# Output: cleaned data
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "SUPER STORE DATA", "cleaned_data.csv")
# Output: charts
CHARTS_PATH = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHARTS_PATH, exist_ok=True)

#LOADING THE DATASET
print("LOADING DATASET...")
print()

df = pd.read_csv(RAW_DATA_PATH, encoding='latin-1')

print(f"Dataset Loaded Successfully")
print(f"Rows : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print()

#Looking at data
print("FIRST LOOK AT RAW DATA")
print()
print(df.head())
print()
print("Column Names:")
print(df.columns.tolist())
print()
print("Data Types:")
print(df.dtypes)
print()

#FIXING DATE COLUMNS
print("FIXING DATE COLUMNS...")
print()

# Converting Order Date and Ship Date from string to real dates
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date']  = pd.to_datetime(df['Ship Date'])

print(f"Order Date type : {df['Order Date'].dtype}")
print(f"Ship Date type  : {df['Ship Date'].dtype}")
print(f"Date Range      : {df['Order Date'].min()} to {df['Order Date'].max()}")
print()

#CHECKING AND HANDLING MISSING VALUES

print("CHECKING MISSING VALUES...")
print()

missing = df.isnull().sum()
print(missing[missing > 0])

# Dropping rows where Sales is missing (critical column)
before = len(df)
df = df.dropna(subset=['Sales'])
after = len(df)
print(f"Removed {before - after} rows with missing Sales")

# Filling missing Postal Code with 00000 (not needed for forecasting)
df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)
print(f"Filled missing Postal Codes with 0")
print()


#REMOVING DUPLICATE ROWS

print("=" * 50)
print("REMOVING DUPLICATES...")
print("=" * 50)

before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"Removed {before - after} duplicate rows")
print(f"Remaining rows: {after}")
print()


#FIXING DATA QUALITY ISSUES

print("=" * 50)
print("FIXING DATA QUALITY...")
print("=" * 50)

# Remove rows where Sales is zero or negative
before = len(df)
df = df[df['Sales'] > 0]
after = len(df)
print(f"Removed {before - after} rows with zero or negative Sales")

# Clean up text columns
df['Region']      = df['Region'].str.strip().str.title()
df['Category']    = df['Category'].str.strip().str.title()
df['Sub-Category']= df['Sub-Category'].str.strip().str.title()
df['Segment']     = df['Segment'].str.strip().str.title()

print(f"Cleaned text columns")
print(f"Regions    : {df['Region'].unique().tolist()}")
print(f"Categories : {df['Category'].unique().tolist()}")
print(f"Segments   : {df['Segment'].unique().tolist()}")
print()


#EXTRACTING DATE FEATURES

print("=" * 50)
print("EXTRACTING DATE FEATURES...")
print("=" * 50)

df['Year']        = df['Order Date'].dt.year
df['Month']       = df['Order Date'].dt.month
df['Quarter']     = df['Order Date'].dt.quarter
df['Day_of_Week'] = df['Order Date'].dt.dayofweek   # 0=Monday, 6=Sunday
df['Week_Number'] = df['Order Date'].dt.isocalendar().week.astype(int)
df['Year_Month']  = df['Order Date'].dt.to_period('M').astype(str)

print(f"Added: Year, Month, Quarter, Day_of_Week(0-Monday,6-Sunday), Week_Number, Year_Month")
print()

# AGGREGATE INTO MONTHLY SALES

print("=" * 50)
print("CREATING MONTHLY SALES SUMMARY...")
print("=" * 50)

monthly_sales = (
    df.groupby('Year_Month')['Sales']
    .sum()
    .reset_index()
)
monthly_sales.columns = ['Month', 'Total_Sales']
monthly_sales['Total_Sales'] = monthly_sales['Total_Sales'].round(2)

print(monthly_sales)
print()

#SAVING CLEANED DATA

print("=" * 50)
print("SAVING CLEANED DATA...")
print("=" * 50)

# Save full cleaned dataset
df.to_csv(PROCESSED_DATA_PATH , index=False)
print(f"Full cleaned data saved to:")
print(f"{PROCESSED_DATA_PATH}")

# Save monthly summary separately
monthly_path = os.path.join(BASE_DIR, "SUPER STORE DATA", "monthly_sales.csv")
monthly_sales.to_csv(monthly_path, index=False)
print(f"Monthly sales summary saved to:")
print(f"{monthly_path}")
print()

#VISUALIZING MONTHLY SALES TREND


print("=" * 50)
print("GENERATING CHART...")
print("=" * 50)

plt.figure(figsize=(14, 6))
plt.plot(
    monthly_sales['Month'],
    monthly_sales['Total_Sales'],
    marker='o',
    color='steelblue',
    linewidth=2,
    markersize=5,
    label='Monthly Sales'
)

plt.title('Monthly Sales Trend (All Years)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save chart
chart_file = os.path.join(CHARTS_PATH, "monthly_sales_trend.png")
plt.savefig(chart_file, dpi=150)
plt.show()

print(f"Chart saved to:")
print(f"{chart_file}")
print()

# FINAL SUMMARY

print("=" * 50)
print("CLEANING COMPLETE — FINAL SUMMARY")
print("=" * 50)
print(f"   Total Orders     : {len(df)}")
print(f"   Date Range       : {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
print(f"   Total Revenue    : ${df['Sales'].sum():,.2f}")
print(f"   Average Order    : ${df['Sales'].mean():,.2f}")
print(f"   Regions          : {df['Region'].nunique()}")
print(f"   Categories       : {df['Category'].nunique()}")
print(f"   Monthly Records  : {len(monthly_sales)}")
print("=" * 50)
