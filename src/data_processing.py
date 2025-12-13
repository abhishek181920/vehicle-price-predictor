import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data(filepath):
    """Load and preprocess the dataset."""
    # Read the CSV file
    df = pd.read_csv(filepath, encoding='latin1')
    
    # Display basic info about the dataset
    print(f"Original dataset shape: {df.shape}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    return df

def preprocess_data(df):
    """Preprocess the data for model training."""
    # Make a copy of the dataframe
    df = df.copy()
    
    # Drop rows with missing price
    if 'price' in df.columns:
        # Convert price to numeric, removing any non-numeric characters
        df['price'] = pd.to_numeric(
            df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True), 
            errors='coerce'
        )
        
        # Remove rows with missing or invalid prices
        initial_count = len(df)
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]  # Remove zero or negative prices
        print(f"Removed {initial_count - len(df)} rows with invalid prices")
    
    # Feature engineering
    current_year = pd.Timestamp.now().year
    df['vehicle_age'] = current_year - df['year']
    
    # Drop unnecessary columns
    cols_to_drop = ['name', 'description']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    # Clean up the engine column to extract engine size (in liters)
    if 'engine' in df.columns:
        df['engine_size'] = df['engine'].str.extract(r'(\d+\.?\d*)\s*[Ll]')
        df['engine_size'] = pd.to_numeric(df['engine_size'], errors='coerce')
    
    # Clean up mileage - remove non-numeric characters and convert to numeric
    if 'mileage' in df.columns:
        df['mileage'] = pd.to_numeric(
            df['mileage'].astype(str).str.replace(r'[^\d]', '', regex=True), 
            errors='coerce'
        )
        # Fill missing mileage with median
        df['mileage'] = df['mileage'].fillna(df['mileage'].median())
    
    # Fill other missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Define features and target
    target_col = 'price'
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Remove any remaining infinite values
        mask = np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Price statistics:\n{y.describe()}")
        
        return X, y
    else:
        return df, None

def get_feature_transformer(X):
    """Create a feature transformer for preprocessing."""
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variable if it's in the numerical features
    if 'price' in numerical_features:
        numerical_features.remove('price')
    
    # Define transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop other columns not specified in the transformers
    )
    
    return preprocessor, numerical_features, categorical_features

def save_preprocessor(preprocessor, filename='models/preprocessor.pkl'):
    """Save the preprocessor to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(preprocessor, filename)
    print(f"Preprocessor saved to {filename}")

def load_preprocessor(filename='models/preprocessor.pkl'):
    """Load the preprocessor from disk."""
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        raise FileNotFoundError(f"Preprocessor file {filename} not found.")

def prepare_data(filepath, test_size=0.2, random_state=42):
    """Prepare the data for training and testing."""
    # Load and preprocess data
    df = load_data(filepath)
    X, y = preprocess_data(df)
    
    if y is None:
        raise ValueError("Target column 'price' not found in the dataset.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Get and fit preprocessor
    preprocessor, numerical_features, categorical_features = get_feature_transformer(X_train)
    
    # Fit and transform the training data
    print("\nFitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    save_preprocessor(preprocessor)
    
    # Print dataset information
    print("\nData preparation complete!")
    print(f"Training set shape: {X_train_processed.shape}")
    print(f"Test set shape: {X_test_processed.shape}")
    print(f"Number of numerical features: {len(numerical_features)}")
    print(f"Number of categorical features: {len(categorical_features)}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Test the data processing pipeline
    data_path = os.path.join('data', 'dataset.csv')
    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path)
        print("\nSample of processed training data:")
        print(X_train[:5])
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
