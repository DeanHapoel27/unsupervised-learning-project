
!pip install igraph
!pip install leidenalg

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.cluster import DBSCAN, MeanShift
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

from sklearn.cluster import AgglomerativeClustering, MeanShift, Birch

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('bodyPerformance.csv')
df.head()
df.shape

for col in df.columns:
  print(f'{col}: {df[col].nunique()}')




def categorize_age(age):
  if 21 <= age <= 33:
    return 1
  elif 36 <= age <= 48:
    return 2
  elif age >= 51:
    return 3
  else:
    return None 

df['AgeCategory'] = df['age'].apply(categorize_age)
new_df = df.copy()
new_df = new_df.dropna(subset=['AgeCategory'])
new_df['AgeCategory'] = new_df['AgeCategory'].astype(int)
print(new_df.head())

new_df.shape


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['class', 'gender', 'AgeCategory'] 

numeric_features = ['height_cm','weight_kg','body fat_%','diastolic',
 'systolic','gripForce','sit and bend forward_cm','sit-ups counts','broad jump_cm',]  # שמות הנומריים

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_features)],
    remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(new_df)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)




results = []

for n_dims in range(2, 15, 3): 
    X_pca = PCA(n_components=n_dims).fit_transform(X_scaled)
    for k in range(2, 9):  
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        results.append({'dims': n_dims, 'k': k, 'silhouette': score})

results_df = pd.DataFrame(results)
heatmap_data = results_df.pivot(index='k', columns='dims', values='silhouette')


plt.figure(figsize=(12, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis')
plt.title('Silhouette Score - K-means')
plt.xlabel('Number of PCA Components')
plt.ylabel('Number of Clusters (k)')
plt.show()


results = []

for n_dims in range(2, 15, 3):  # לדוגמה: 2, 5, 8, 11, 14
    X_pca = PCA(n_components=n_dims).fit_transform(X_scaled)

    # Hierarchical (Agglomerative)
    for k in range(2, 9):
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        results.append({'dims': n_dims, 'k': k, 'silhouette': score, 'method': 'Hierarchical'})

    # Birch
    for k in range(2, 9):
        model = Birch(n_clusters=k)
        labels = model.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        results.append({'dims': n_dims, 'k': k, 'silhouette': score, 'method': 'Birch'})


results_df = pd.DataFrame(results)

for method in results_df['method'].unique():
    method_df = results_df[results_df['method'] == method]
    heatmap_data = method_df.pivot(index='k', columns='dims', values='silhouette')

    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='plasma')
    plt.title(f'Silhouette Score - {method}')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Number of Clusters (k)')
    plt.show()



for n_dims in range(2, 15, 3): 
    X_pca = PCA(n_components=n_dims).fit_transform(X_scaled)
    for k in range(2, 9):  
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        results.append({'dims': n_dims, 'k': k, 'silhouette': score})


results_df = pd.DataFrame(results)
heatmap_data = results_df.pivot(index='k', columns='dims', values='silhouette')

plt.figure(figsize=(12, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis')
plt.title('Silhouette Score - GMM')
plt.xlabel('Number of PCA Components')
plt.ylabel('Number of Clusters (k)')
plt.show()



np.random.shuffle(X_scaled)
group_size = len(X_scaled) // 25
groups = []
for i in range(25):
    start_index = i * group_size
    end_index = (i + 1) * group_size if i < 14 else len(X_scaled)  # Handle the last group
    groups.append(X_scaled[start_index:end_index])


import matplotlib.pyplot as plt

# Assuming 'groups' and 'X_scaled' are defined from the previous code block

results = []
# KMeans
for group in groups:
    X_pca = PCA(n_components=2).fit_transform(group)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    results.append({'method': 'KMeans', 'silhouette': score})

# BIRCH
for group in groups:
    X_pca = PCA(n_components=2).fit_transform(group)
    birch = Birch(n_clusters=4)
    labels = birch.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    results.append({'method': 'Birch', 'silhouette': score})

# GMM
for group in groups:
    X_pca = PCA(n_components=2).fit_transform(group)
    gmm = GaussianMixture(n_components=4, random_state=42)
    labels = gmm.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    results.append({'method': 'GMM', 'silhouette': score})


for group in groups:
    X_pca = PCA(n_components=2).fit_transform(group)
    agg_clustering = AgglomerativeClustering(n_clusters=2)
    labels = agg_clustering.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    results.append({'method': 'Hierarchical', 'silhouette': score})

results_df = pd.DataFrame(results)


method_stats = results_df.groupby('method')['silhouette'].agg(['mean', 'std'])


plt.figure(figsize=(12, 4))
colors = {'KMeans': 'blue', 'Birch': 'green', 'GMM': 'red', 'Hierarchical': 'purple'}

for method, row in method_stats.iterrows():
  plt.bar(method, row['mean'], yerr=row['std'], color=colors[method], capsize=5)

plt.xlabel("Clustering Algorithm")
plt.ylabel("Average Silhouette Score")
plt.title("Comparison of Clustering Algorithms")
plt.show()



method_stats = results_df.groupby('method')['silhouette'].agg(['mean', 'std'])
method_stats


from scipy import stats


kmeans_scores = results_df[results_df['method'] == 'KMeans']['silhouette']
gmm_scores = results_df[results_df['method'] == 'GMM']['silhouette']
hierarchical_scores = results_df[results_df['method'] == 'Hierarchical']['silhouette']


fvalue, pvalue = stats.f_oneway(kmeans_scores, gmm_scores, hierarchical_scores)
print(f"ANOVA p-value: {pvalue}")


def perform_t_test(group1, group2):
  t_statistic, p_value = stats.ttest_ind(group1, group2)
  return p_value


p_values = {
    'KMeans vs GMM': perform_t_test(kmeans_scores, gmm_scores),
    'KMeans vs Hierarchical': perform_t_test(kmeans_scores, hierarchical_scores),
    'GMM vs Hierarchical': perform_t_test(gmm_scores, hierarchical_scores)
}

p_value_df = pd.DataFrame(list(p_values.items()), columns=['Comparison', 'p-value'])
print("\nPairwise t-test p-values:")
p_value_df



birch_scores = results_df[results_df['method'] == 'Birch']['silhouette']
kmeans_scores = results_df[results_df['method'] == 'KMeans']['silhouette']
gmm_scores = results_df[results_df['method'] == 'GMM']['silhouette']
hierarchical_scores = results_df[results_df['method'] == 'Hierarchical']['silhouette']


def perform_t_test(group1, group2):
  t_statistic, p_value = stats.ttest_ind(group1, group2)
  return p_value


p_values = {
    'Birch vs KMeans': perform_t_test(birch_scores, kmeans_scores),
    'Birch vs GMM': perform_t_test(birch_scores, gmm_scores),
    'Birch vs Hierarchical': perform_t_test(birch_scores, hierarchical_scores),
}


p_value_df = pd.DataFrame(list(p_values.items()), columns=['Comparison', 'p-value'])
print("\nPairwise t-test p-values:")
p_value_df




import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


np.random.shuffle(X_scaled)
group_size = len(X_scaled) // 20
groups = [X_scaled[i * group_size:(i + 1) * group_size] for i in range(20)]


categorical_indices = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat'][0]
ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
num_ohe_features = len(ohe_features)
categorical_data = X_scaled[:, :num_ohe_features]


mutual_info_results = []
for i, group in enumerate(groups):
  group_results = {}
  X_pca = PCA(n_components=2).fit_transform(group)
  

  kmeans = KMeans(n_clusters=4, random_state=42)
  kmeans_labels = kmeans.fit_predict(X_pca)
  for j, feature in enumerate(categorical_features):
    mi = mutual_info_score(kmeans_labels, categorical_data[i*group_size:(i+1)*group_size, j])
    group_results[f'k-means_{feature}'] = mi


  gmm = GaussianMixture(n_components=4, random_state=42)
  gmm_labels = gmm.fit_predict(X_pca)
  for j, feature in enumerate(categorical_features):
    mi = mutual_info_score(gmm_labels, categorical_data[i*group_size:(i+1)*group_size, j])
    group_results[f'GMM_{feature}'] = mi


  hierarchical = AgglomerativeClustering(n_clusters=2)
  hierarchical_labels = hierarchical.fit_predict(X_pca)
  for j, feature in enumerate(categorical_features):
      mi = mutual_info_score(hierarchical_labels, categorical_data[i*group_size:(i+1)*group_size, j])
      group_results[f'Hierarchical_{feature}'] = mi

  mutual_info_results.append(group_results)


mutual_info_df = pd.DataFrame(mutual_info_results)
mi_means = mutual_info_df.mean()
mi_stds = mutual_info_df.std()


plt.figure(figsize=(10, 6))
plt.bar(mi_means.index, mi_means.values, yerr=mi_stds.values, capsize=5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mutual Information')
plt.title('Average Mutual Information between Clusters and Categorical Features')
plt.tight_layout()
plt.show()


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



np.random.shuffle(X_scaled)
group_size = len(X_scaled) // 20
groups = [X_scaled[i * group_size:(i + 1) * group_size] for i in range(20)]


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
