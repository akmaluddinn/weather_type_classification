import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load dataset"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocessing data"""
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df_cleaned = df[(z_scores < 3).all(axis=1)]
    
    # Separate features and target
    X = df_cleaned.drop('Weather Type', axis=1)
    y = df_cleaned['Weather Type']
    
    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, preprocessor