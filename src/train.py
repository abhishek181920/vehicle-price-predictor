import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split
import json

from data_processing import prepare_data, load_preprocessor

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("husl")

def train_model(X_train, y_train, X_val=None, y_val=None, use_early_stopping=True):
    """Train the XGBoost model with optimal hyperparameters."""
    # Prepare evaluation set if validation data is provided
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        print("Using validation set for early stopping")
    
    model_params = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'n_jobs': -1
    }
    
    # Add early stopping if validation set is provided
    if use_early_stopping and eval_set is not None:
        model_params['early_stopping_rounds'] = 50
        model_params['eval_metric'] = 'rmse'
    
    model = XGBRegressor(**model_params)
    
    print("Starting model training...")
    
    # Prepare fit parameters
    fit_params = {
        'verbose': 100
    }
    
    if eval_set is not None:
        fit_params['eval_set'] = eval_set
    
    model.fit(X_train, y_train, **fit_params)
    
    # Print feature importance
    print("\nFeature importance:")
    importance = model.feature_importances_
    for i, imp in enumerate(importance[:10]):  # Show top 10 features
        print(f"  Feature {i}: {imp:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return metrics, y_pred

def plot_learning_curve(estimator, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """Generate a learning curve plot with custom handling for XGBoost."""
    try:
        # Create a custom scoring function that works with XGBoost
        def xgb_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)  # Negative because we want to maximize
        
        # Create a copy of the estimator without callbacks to avoid early stopping issues
        import copy
        est = copy.deepcopy(estimator)
        if hasattr(est, 'set_params'):
            est.set_params(**{'early_stopping_rounds': None, 'callbacks': None})
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            est, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
            scoring=xgb_scorer, random_state=42, verbose=1
        )
        
        # Convert to positive MSE
        train_scores = -train_scores
        test_scores = -test_scores
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training error')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation error')
        plt.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1, color='r')
        plt.fill_between(train_sizes, 
                        np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                        np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                        alpha=0.1, color='g')
        
        plt.xlabel('Training examples')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the plot
        os.makedirs('models/plots', exist_ok=True)
        plt.savefig('models/plots/learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_feature_importance(model, feature_names, max_num_features=20):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:max_num_features]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('models/plots/feature_importance.png')
    plt.close()

def save_metrics(metrics, filename='models/metrics.json'):
    """Save evaluation metrics to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    print("Starting model training...")
    
    # Prepare data
    data_path = os.path.join('data', 'dataset.csv')
    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path)
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        return
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model with validation set for early stopping
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print("\nModel Performance on Test Set:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Generate and save plots
    print("\nGenerating plots...")
    plot_learning_curve(model, X_train, y_train)
    
    # Get feature names from the preprocessor
    try:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                # Get feature names from one-hot encoded features
                feature_names.extend(trans.named_steps['onehot'].get_feature_names_out(cols).tolist())
            elif hasattr(trans, 'get_feature_names_out'):
                feature_names.extend(trans.get_feature_names_out(cols).tolist())
            else:
                feature_names.extend(cols)
        
        # Plot feature importance if we have feature names
        if feature_names and len(feature_names) > 0:
            plot_feature_importance(model, feature_names)
            print("\nGenerated feature importance plot.")
    except Exception as e:
        print(f"\nCould not generate feature importance plot: {str(e)}")
    
    # Save model and metrics
    print("\nSaving model and metrics...")
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/vehicle_price_predictor.pkl'
    joblib.dump({
        'model': model,
        'preprocessor': preprocessor,
        'metrics': metrics
    }, model_path)
    
    # Save metrics separately as well
    save_metrics(metrics)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {os.path.abspath(model_path)}")
    print(f"\nModel Performance Summary:")
    print(f"- RMSE: ${metrics['rmse']:,.2f}")
    print(f"- MAE: ${metrics['mae']:,.2f}")
    print(f"- R² Score: {metrics['r2']:.4f}")
    print(f"- MAPE: {metrics['mape']:.2f}%")
    
    # Create a simple example prediction
    try:
        sample_idx = 0
        sample = X_test[sample_idx].reshape(1, -1)
        pred = model.predict(sample)[0]
        actual = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
        print(f"\nExample Prediction:")
        print(f"- Predicted Price: ${pred:,.2f}")
        print(f"- Actual Price: ${actual:,.2f}")
        print(f"- Difference: ${abs(pred - actual):,.2f}")
    except Exception as e:
        print(f"\nCould not generate example prediction: {str(e)}")

if __name__ == "__main__":
    main()
