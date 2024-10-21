
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Paths to the model, dataset, and mapping files
model_path = "/personal/PCE_model_full_perovskite.joblib"  # Model used for prediction and trained on encoded data
data_path = "/personal/encoded_experiment_data.csv"  # Dataset already encoded and used during model training
mapping_path = "/personal/target_PCEencoder_mapping.joblib"  # Mapping for converting encoded values back to original categories

# Load the trained model and the encoding mappings
print("Loading model from", model_path)
model = joblib.load(model_path)
print("Model loaded successfully.")

print("Loading mapping from", mapping_path)
mapping = joblib.load(mapping_path)
print("Mapping loaded successfully.")

# Reverse the mapping to decode from encoded values back to original values
reversed_mapping = {feature: {v: k for k, v in feature_mapping.items()} for feature, feature_mapping in mapping.items()}

# Load the dataset (this dataset was used during model training, hence it is encoded)
print("Loading data from", data_path)
data = pd.read_csv(data_path)
print("Data loaded successfully.")

# Get the list of features used in the trained model
model_features = list(model.get_booster().feature_names)

# Ensure the dataset contains all the features that were used during model training
if not all(feature in data.columns for feature in model_features):
    raise ValueError("Not all model features are present in the dataset. Please check if the dataset is consistent with the model training.")

# Perform SHAP analysis and optimization suggestions for each type of Perovskite Solar Cell (PSC)
psc_types = data['Type'].unique()
for psc_type in psc_types:
    print(f"\nAnalyzing PSC Type: {psc_type}")
    subset = data[data['Type'] == psc_type]

    # Perform SHAP analysis using all features in the model
    explainer = shap.Explainer(model)
    shap_values = explainer(subset[model_features])

    # Calculate feature importance using SHAP values
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame([model_features, shap_sum.tolist()]).T
    importance_df.columns = ['feature', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    # Filter out certain features that are not of interest
    filtered_importance_df = importance_df[~importance_df['feature'].isin(['Cell_stack_sequence', 'Perovskite_composition_long_form'])]

    # Plot the top 5 important features for the current PSC type
    top_features_df = filtered_importance_df.head(5)
    plt.figure(figsize=(10, 4))
    plt.bar(top_features_df['feature'], top_features_df['shap_importance'], width=0.4)
    plt.ylabel('SHAP Importance')
    plt.title(f"Top 5 Important Features for PSC Type {psc_type} (Filtered)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Provide optimization suggestions for each of the top features
    print(f"Optimization suggestions for PSC Type {psc_type}:")
    for feature in top_features_df['feature']:
        # Evaluate different category values in the original dataset (not using encoded values directly)
        category_values = data[feature].unique()
        best_score = -np.inf
        best_category = None

        # Simulate predictions by altering one feature at a time and checking the impact
        for category in category_values:
            modified_subset = subset.copy()
            modified_subset[feature] = category
            prediction = model.predict(modified_subset[model_features])
            score = prediction.mean()

            if score > best_score:
                best_score = score
                best_category = category

        # Get the original and optimized category using the mapping
        if not subset[feature].empty:
            original_category = subset[feature].mode()[0] if not subset[feature].mode().empty else None
        else:
            original_category = None

        if original_category is not None and feature in reversed_mapping:
            original_mapped_value = reversed_mapping[feature].get(original_category, f"No mapping for {original_category}")
            best_mapped_value = reversed_mapping[feature].get(best_category, f"No mapping for {best_category}")
        else:
            original_mapped_value = "No mapping for feature"
            best_mapped_value = "No mapping for feature"

        print(f"  - {feature}: Optimal value = {best_mapped_value} (Original: {original_mapped_value})")

print("Analysis completed.")
