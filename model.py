import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Popularity_Score'] = df['Popularity_Score'].fillna(df['Popularity_Score'].median())
    df['Description'] = df['Description'].fillna('No Description')
    df['Creation_Date'] = pd.to_datetime(df['Creation_Date'], errors='coerce')
    df['Creation_Date'] = df['Creation_Date'].fillna(pd.Timestamp.now())
    df['Year'] = df['Creation_Date'].dt.year
    df['Month'] = df['Creation_Date'].dt.month
    return df

# Train models
def train_models(df):
    X = df[['Popularity_Score', 'Year', 'Month']]
    y = df['Popularity_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return lr_model, rf_model

# Predict function
def predict_popularity(model, popularity_score, year, month):
    X_new = np.array([[popularity_score, year, month]])
    return model.predict(X_new)[0]

# For Flask: load data and train models once
csv_path = 'art.csv'
df = load_data(csv_path)
lr_model, rf_model = train_models(df)
