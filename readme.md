# Sales and Demand Forecasting System

A machine learning project that predicts future sales using historical data from the Superstore dataset. Built with Python and Scikit-learn to help businesses make data-driven decisions around inventory, staffing, and cash flow planning.

---

## Project Overview

This project builds a complete end-to-end sales forecasting pipeline that:

- Cleans and prepares historical sales data
- Engineers time-based features such as lag values, rolling averages, and seasonality flags
- Trains and compares Linear Regression and Random Forest models
- Forecasts the next 6 months of sales
- Generates business-friendly visualizations and a written business report

---

## Project Structure

```
SALES AND DEMAND FORECASTING/
│
├── models/
│   ├── linear_regression_model.pkl    # Trained Linear Regression model
│   ├── random_forest_model.pkl        # Trained Random Forest model
│   └── scaler.pkl                     # StandardScaler for feature scaling
│
├── outputs/
│   ├── charts/
│   │   ├── actual_vs_predicted.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_importance.png
│   │   ├── forecast_dashboard.png
│   │   ├── forecast_future.png
│   │   ├── monthly_sales_trend.png
│   │   ├── sales_by_category.png
│   │   ├── sales_by_month.png
│   │   ├── sales_by_quarter.png
│   │   ├── sales_by_region.png
│   │   ├── sales_by_season.png
│   │   ├── sales_trend_overview.png
│   │   └── yearly_growth.png
│   └── reports/
│       └── business_report.txt        # Auto-generated business report
│
├── src/
│   ├── data_cleaning.py               # Step 1: Load and clean raw data
│   ├── feature_engineering.py         # Step 2: Create time-based features
│   ├── model_training.py              # Step 3: Train models and forecast
│   └── visualization.py              # Step 4: Generate charts and report
│
├── SUPER STORE DATA/
│   ├── Sample - Superstore.csv        # Raw dataset (not pushed to GitHub)
│   ├── cleaned_data.csv               # Cleaned dataset output
│   ├── featured_data.csv              # Feature engineered dataset output
│   ├── monthly_sales.csv              # Monthly aggregated sales
│   └── load_data.py                   # Initial data exploration script
│
├── venv/                              # Virtual environment (not pushed)
├── requirements.txt                   # Required Python libraries
└── .gitignore                         # Files excluded from Git
```

---

## Dataset

**Superstore Sales Dataset** from Kaggle

- 9,994 order records
- Date range: 2014 to 2017
- Features: Order Date, Region, Category, Sub-Category, Sales, Quantity, Profit

Download the dataset here:
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

Place the downloaded file at:
```
SUPER STORE DATA/Sample - Superstore.csv
```

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/masihatasneem99-png/FUTURE_ML_01.git
cd sales-demand-forecasting
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```

---

## How to Run

Run each script in order from the project root:

### Step 0: Explore the Data (Optional)
```bash
python "SUPER STORE DATA/load_data.py"
```
Loads the raw dataset and prints shape, column names, data types, missing values, and basic statistics. Does not save or change anything.

### Step 1: Data Cleaning
```bash
python src/data_cleaning.py
```
Loads the raw CSV, fixes date formats, removes duplicates, handles missing values, extracts date features, aggregates monthly sales, and saves the cleaned dataset.

**Output files:**
- `SUPER STORE DATA/cleaned_data.csv`
- `SUPER STORE DATA/monthly_sales.csv`
- `outputs/charts/monthly_sales_trend.png`

### Step 2: Feature Engineering
```bash
python src/feature_engineering.py
```
Creates lag features, rolling averages, seasonality flags, holiday indicators, and growth rate features. Saves the featured dataset and generates exploratory charts.

**Output files:**
- `SUPER STORE DATA/featured_data.csv`
- `outputs/charts/sales_by_month.png`
- `outputs/charts/sales_by_quarter.png`
- `outputs/charts/sales_by_season.png`
- `outputs/charts/correlation_heatmap.png`

### Step 3: Model Training
```bash
python src/model_training.py
```
Trains Linear Regression and Random Forest models, evaluates both, picks the best performer, saves trained models, and forecasts the next 6 months.

**Output files:**
- `models/linear_regression_model.pkl`
- `models/random_forest_model.pkl`
- `models/scaler.pkl`
- `outputs/charts/actual_vs_predicted.png`
- `outputs/charts/feature_importance.png`
- `outputs/charts/forecast_future.png`

### Step 4: Visualization and Report
```bash
python src/visualization.py
```
Generates the full forecast dashboard, regional and category breakdowns, yearly growth charts, and saves the complete business report.

**Output files:**
- `outputs/charts/sales_trend_overview.png`
- `outputs/charts/sales_by_region.png`
- `outputs/charts/sales_by_category.png`
- `outputs/charts/yearly_growth.png`
- `outputs/charts/forecast_dashboard.png`
- `outputs/reports/business_report.txt`

---

## Features Engineered

| Feature | Description |
|---|---|
| Year, Month, Quarter | Basic date components |
| Season | Winter, Spring, Summer, Fall |
| Is_Holiday_Season | 1 if November or December |
| Is_Peak_Month | 1 if March, April, or September |
| Is_Quarter_End | 1 if end of quarter month |
| Lag_1, Lag_2, Lag_3 | Sales from previous months |
| Lag_12 | Sales from same month last year |
| Rolling_3, Rolling_6, Rolling_12 | Moving averages |
| Month_Growth_Rate | Month over month growth percentage |

---

## Model Performance

Two models are trained and compared automatically:

| Model | Description |
|---|---|
| Linear Regression | Baseline model — simple and interpretable |
| Random Forest | Advanced model — handles seasonality and non-linear patterns |

**Evaluation Metrics:**

| Metric | Description |
|---|---|
| R² Score | How well the model explains sales variation — closer to 1.0 is better |
| MAE | Average dollar error per prediction |
| RMSE | Penalizes larger errors more heavily |
| MAPE | Error as a percentage of actual sales — below 10% is excellent |

The best performing model is automatically selected for the 6-month forecast.

---

## Visualizations Generated

| Chart | What It Shows |
|---|---|
| monthly_sales_trend.png | Overall sales trend across all years |
| sales_by_month.png | Average sales for each month |
| sales_by_quarter.png | Average sales per quarter |
| sales_by_season.png | Sales broken down by season |
| correlation_heatmap.png | Relationships between all features |
| actual_vs_predicted.png | Model predictions vs real sales |
| feature_importance.png | Which features drive the model most |
| forecast_future.png | Next 6 months forecast line chart |
| sales_by_region.png | Regional sales breakdown |
| sales_by_category.png | Category sales breakdown |
| yearly_growth.png | Year over year growth |
| sales_trend_overview.png | Full historical and forecast overview |
| forecast_dashboard.png | Complete business dashboard |

---

## Business Report

After running `visualization.py`, a complete business report is saved to:
```
outputs/reports/business_report.txt
```

The report includes:
- Business overview and key metrics
- Model performance summary
- Month by month 6-month forecast
- Inventory, staffing, and marketing recommendations
- Key business insights

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
notebook
ipykernel
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning models |
| Matplotlib | Data visualization |
| Seaborn | Statistical charts |
| VS Code | Development environment |
| Git | Version control |

---

## Author

**Masiha Tasneem**
- GitHub: https://github.com/masihatasneem99-png
- LinkedIn: www.linkedin.com/in/masihatasneem

---

## License

This project is open source and available under the MIT License.