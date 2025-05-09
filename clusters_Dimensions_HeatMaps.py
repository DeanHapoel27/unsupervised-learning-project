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




def run_clusters_dimensions_heatmaps():
    
    //K-means
    
    results = []
    for n_dims in range(2, 15, 3):  # 2 עד 12 ממדים
        X_pca = PCA(n_components=n_dims).fit_transform(X_scaled)
        for k in range(2, 9):  # 2 עד 10 קלאסטרים
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
    
    
    // Hierarchical, Birch
    
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
    
    \\GMM
    from sklearn.mixture import GaussianMixture
    results = []
    
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


if __name__ == "__main__":
    run_clusters_dimensions_heatmaps()
    
