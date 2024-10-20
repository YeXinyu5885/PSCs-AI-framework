import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import argparse

# Custom R^2 scoring function with progress monitoring
def custom_r2_scorer(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    print(f"Evaluating model... Score: {score:.4f} - Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return score

# Main function to handle the model training and evaluation
def main(input_file, output_file):
    # Setting seed for reproducibility
    np.random.seed(61)

    # Reading the dataset from a user-specified CSV file
    try:
        # Attempt to read using UTF-8 encoding
        data = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, attempt to read using GBK encoding
            data = pd.read_csv(input_file, encoding='gbk')
        except UnicodeDecodeError:
            # If GBK fails, attempt to read using UTF-16 encoding
            data = pd.read_csv(input_file, encoding='utf-16')

    # Drop columns where all values are NaN
    data = data.dropna(axis=1, how='all')

    # Fill NaN values with the mean of the column
    data = data.fillna(data.mean())

    # Target column
    target = 'JV_default_Jsc'

    # Splitting features and target
    features = data.columns[:-1].tolist()
    X = data[features]
    y = data[target]

    # Initialize TargetEncoder
    encoder = TargetEncoder()

    # Target encoding for categorical columns
    for column in X.columns:
        if X[column].dtype == object:  # If the column is categorical
            X[column] = encoder.fit_transform(X[column], y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=54)

    # XGBoost model with GridSearchCV to find optimal hyperparameters
    param_grid = {
        'n_estimators': [100, 300, 500, 700, 900],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.3, 0.5, 0.7]
    }

    # Initialize the XGBoost regressor
    xgb_model = XGBRegressor()

    # Perform grid search with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, scoring='r2', verbose=1, n_jobs=-1)
    
    # Fit model
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Evaluate performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Output the performance metrics
    print('Best Parameters from Grid Search:', grid_search.best_params_)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Error (MAE):', mae)
    print('R^2 (Coefficient of Determination):', r2)

    # Save predictions to a user-specified file
    pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv(output_file, index=False)
    
if __name__ == "__main__":
    # Argument parser for user-specified input/output files
    parser = argparse.ArgumentParser(description='Train XGBoost model on input data and save predictions.')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', required=True, help='Path to save the output predictions CSV file')
    
    args = parser.parse_args()

    # Run the main function
    main(args.input, args.output)
