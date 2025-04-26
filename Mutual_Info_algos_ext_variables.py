import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import ConvexHull

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, Birch
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import igraph
import leidenalg

def run_mutual_info_algos_ext_variables():

  # Split data into 20 groups
  np.random.shuffle(X_scaled)
  group_size = len(X_scaled) // 20
  groups = [X_scaled[i * group_size:(i + 1) * group_size] for i in range(20)]
  
  
  # Extract one-hot encoded categorical features
  categorical_indices = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat'][0]
  ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
  num_ohe_features = len(ohe_features)
  categorical_data = X_scaled[:, :num_ohe_features]
  
  
  mutual_info_results = []
  for i, group in enumerate(groups):
    group_results = {}
    X_pca = PCA(n_components=2).fit_transform(group)
    
    # k-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca)
    for j, feature in enumerate(categorical_features):
      mi = mutual_info_score(kmeans_labels, categorical_data[i*group_size:(i+1)*group_size, j])
      group_results[f'k-means_{feature}'] = mi
  
    # GMM
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca)
    for j, feature in enumerate(categorical_features):
      mi = mutual_info_score(gmm_labels, categorical_data[i*group_size:(i+1)*group_size, j])
      group_results[f'GMM_{feature}'] = mi
  
    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=2)
    hierarchical_labels = hierarchical.fit_predict(X_pca)
    for j, feature in enumerate(categorical_features):
        mi = mutual_info_score(hierarchical_labels, categorical_data[i*group_size:(i+1)*group_size, j])
        group_results[f'Hierarchical_{feature}'] = mi
  
    mutual_info_results.append(group_results)
  
  
  mutual_info_df = pd.DataFrame(mutual_info_results)
  mi_means = mutual_info_df.mean()
  mi_stds = mutual_info_df.std()
  
  
  plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
  
  bar_width = 0.2
  index = np.arange(len(categorical_features))
  
  colors = {'k-means': 'blue', 'GMM': 'red', 'Hierarchical': 'purple'}
  
  for i, algorithm in enumerate(['k-means', 'GMM', 'Hierarchical']):
      mi_values = [mi_means[f'{algorithm}_{feature}'] for feature in categorical_features]
      mi_errors = [mi_stds[f'{algorithm}_{feature}'] for feature in categorical_features]
      plt.bar(index + i * bar_width, mi_values, bar_width, label=algorithm.capitalize(), yerr=mi_errors, capsize=5, color=colors[algorithm])
  
  plt.xlabel('Categorical Features')
  plt.ylabel('Mutual Information')
  plt.title('Mutual Information between Clusters and Categorical Features')
  plt.xticks(index + bar_width, categorical_features)
  plt.legend()
  plt.tight_layout()  # Adjust layout to prevent labels from overlapping
  plt.show()
  
  
  
  
  
  #amount of clusters acorrding to the amount of uniqe values of every external variable
  
  # Split data into 20 groups
  np.random.shuffle(X_scaled)
  group_size = len(X_scaled) // 20
  groups = [X_scaled[i * group_size:(i + 1) * group_size] for i in range(20)]
  
  # Extract one-hot encoded categorical features
  categorical_indices = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat'][0]
  ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
  num_ohe_features = len(ohe_features)
  categorical_data = X_scaled[:, :num_ohe_features]
  
  
  mutual_info_results = []
  for i, group in enumerate(groups):
      group_results = {}
      X_pca = PCA(n_components=2).fit_transform(group)
  
      # k-means
      kmeans = KMeans(n_clusters=4, random_state=42) # Use different n_clusters for each categorical feature
      kmeans_labels = kmeans.fit_predict(X_pca)
      for j, feature in enumerate(categorical_features):
          # Determine the number of unique values for the current categorical feature in the group
          unique_vals = np.unique(new_df[feature][i*group_size:(i+1)*group_size])
          n_clusters_feature = len(unique_vals)
          kmeans_feature = KMeans(n_clusters=min(n_clusters_feature, 4), random_state=42) # Use unique values as n_clusters
          kmeans_labels = kmeans_feature.fit_predict(X_pca)
          mi = mutual_info_score(kmeans_labels, categorical_data[i*group_size:(i+1)*group_size, j])
          group_results[f'k-means_{feature}'] = mi
  
      # Add similar code for GMM and Hierarchical clustering, using adjusted n_components or n_clusters
      
      # GMM
      for j, feature in enumerate(categorical_features):
          unique_vals = np.unique(new_df[feature][i*group_size:(i+1)*group_size])
          n_components_feature = len(unique_vals)
          gmm = GaussianMixture(n_components=min(n_components_feature, 4), random_state=42)
          gmm_labels = gmm.fit_predict(X_pca)
          mi = mutual_info_score(gmm_labels, categorical_data[i*group_size:(i+1)*group_size, j])
          group_results[f'GMM_{feature}'] = mi
  
      # Hierarchical clustering
      for j, feature in enumerate(categorical_features):
          unique_vals = np.unique(new_df[feature][i*group_size:(i+1)*group_size])
          n_clusters_feature = len(unique_vals)
          hierarchical = AgglomerativeClustering(n_clusters=min(n_clusters_feature, 4))
          hierarchical_labels = hierarchical.fit_predict(X_pca)
          mi = mutual_info_score(hierarchical_labels, categorical_data[i*group_size:(i+1)*group_size, j])
          group_results[f'Hierarchical_{feature}'] = mi
  
      mutual_info_results.append(group_results)
  
  mutual_info_df = pd.DataFrame(mutual_info_results)
  mi_means = mutual_info_df.mean()
  mi_stds = mutual_info_df.std()
  
  plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
  
  bar_width = 0.2
  index = np.arange(len(categorical_features))
  
  colors = {'k-means': 'blue', 'GMM': 'red', 'Hierarchical': 'purple'}
  
  for i, algorithm in enumerate(['k-means', 'GMM', 'Hierarchical']):
      mi_values = [mi_means[f'{algorithm}_{feature}'] for feature in categorical_features]
      mi_errors = [mi_stds[f'{algorithm}_{feature}'] for feature in categorical_features]
      plt.bar(index + i * bar_width, mi_values, bar_width, label=algorithm.capitalize(), yerr=mi_errors, capsize=5, color=colors[algorithm])
  
  plt.xlabel('Categorical Features')
  plt.ylabel('Mutual Information')
  plt.title('Mutual Information between Clusters and Categorical Features')
  plt.xticks(index + bar_width, categorical_features)
  plt.legend()
  plt.tight_layout()  # Adjust layout to prevent labels from overlapping
  plt.show()


if __name__ == "__main__":
  run_mutual_info_algos_ext_variables()
    
