
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("D:\SALES AND DEMAND FORECASTING\SUPER STORE DATA\Sample - Superstore.csv", encoding='latin-1')

print("Dataset Shape:", df.shape)
print()

print("First 5 Rows:")
print(df.head())
print()

print("Column Names:")
print(df.columns.tolist())
print()

print("Data Types:")
print(df.dtypes)
print()

print("Missing Values:")
print(df.isnull().sum())
print()

print("Sales Column Statistics:")
print(df['Sales'].describe())