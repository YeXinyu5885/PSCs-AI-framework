
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (renamed to be more descriptive)
file_path = '/personal/filtered_data_for_clustering.csv'
data = pd.read_csv(file_path)

# Store the features that are not used in clustering but will be used later
excluded_features = data[['JV_default_PCE', 'ETL_stack_sequence', 'HTL_stack_sequence','Backcontact_stack_sequence']]

# Remove the excluded features from the clustering data
data_for_clustering = data.drop(['JV_default_PCE', 'ETL_stack_sequence', 'HTL_stack_sequence','Backcontact_stack_sequence'], axis=1)

# Standardize the data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering.select_dtypes(include=['float64', 'int', 'bool']))

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
data_tsne = tsne.fit_transform(data_scaled)

# Apply K-Means clustering
k = 6  # Choose the number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(data_tsne)

# Visualize the t-SNE result with clusters
plt.figure(figsize=(8, 6))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=clusters, cmap='viridis')
plt.title("t-SNE Visualization with Clustered Data")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar()  # Show the color bar for cluster labels
plt.show()

# Add cluster labels to the original dataset
data['cluster_label'] = clusters

# Create the output directory if it does not exist
output_dir = '/personal/CTL-STACK/'
os.makedirs(output_dir, exist_ok=True)

# Save the clustered samples to separate CSV files based on cluster labels
for i in range(k):
    print(f"Processing cluster {i}")  # Print the cluster being processed
    cluster_data = data[data['cluster_label'] == i]
    
    # Save each cluster's data to a separate CSV file
    filename = f'/personal/CTL-STACK/cluster_{i}_samples.csv'
    print(f"Saving to {filename}")  # Print the file being saved
    cluster_data.to_csv(filename, index=False)

# Now load one specific cluster's data for further analysis

# Specify the cluster number you want to analyze
cluster_number = 0  # Replace with the actual cluster number you want to analyze

# Load the specific cluster data file
file_path = f'/personal/CTL-STACK/cluster_{cluster_number}_samples.csv'
data = pd.read_csv(file_path)

# Group by multiple sequence columns and calculate mean JV_default_PCE values
grouped_data = data.groupby(['ETL_stack_sequence', 'HTL_stack_sequence', 'Backcontact_stack_sequence'])['JV_default_PCE'].mean().reset_index()

# Select the top 5 combinations with the highest average JV_default_PCE values
top_combinations = grouped_data.nlargest(5, 'JV_default_PCE')

# Set up categories for the radar chart
categories = ['ETL_stack_sequence', 'HTL_stack_sequence', 'Backcontact_stack_sequence', 'JV_default_PCE']
N = len(categories)

# Compute the angle for the radar chart's axes
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the circle

# Initialize radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Set the number of concentric circles and the values for them
radar_center = 14
num_circles = 8
circle_values = [radar_center + i for i in range(num_circles + 1)]
ax.set_ylim(radar_center, radar_center + num_circles)
ax.set_yticks(circle_values)

# Plot the radar chart with top combinations
for index, row in top_combinations.iterrows():
    values = row[categories].tolist()
    values += values[:1]  # Close the plot
    label = f"Comb {index + 1}: " + ', '.join(f"{cat}: {row[cat]:.6f}" for cat in categories)
    ax.plot(angles, values, 'o-', linewidth=2, label=label)
    ax.fill(angles, values, alpha=0.4)  # Fill with transparency

# Label the axes with category names
ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Set the title and display the radar chart
plt.title('Top 5 Combinations Radar Chart')
plt.show()
