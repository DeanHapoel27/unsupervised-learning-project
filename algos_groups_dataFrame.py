import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def run_algos_groups_dataframe():
    #take 25 different and equal groups from data base
    
    np.random.shuffle(X_scaled)
    
    group_size = len(X_scaled) // 25
    
    groups = []
    for i in range(25):
        start_index = i * group_size
        end_index = (i + 1) * group_size if i < 14 else len(X_scaled)
        groups.append(X_scaled[start_index:end_index])
    
    
    
    
    
    #Algorithm runs on the groups, measuring results, and presenting them in a graph.
    
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
    
    # Hierarchical Clustering
    for group in groups:
        X_pca = PCA(n_components=2).fit_transform(group)
        agg_clustering = AgglomerativeClustering(n_clusters=2)
        labels = agg_clustering.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        results.append({'method': 'Hierarchical', 'silhouette': score})
    
    results_df = pd.DataFrame(results)
    
    # Calculate mean and standard deviation for each method
    method_stats = results_df.groupby('method')['silhouette'].agg(['mean', 'std'])
    
    # Create the plot
    plt.figure(figsize=(12, 4))
    colors = {'KMeans': 'blue', 'Birch': 'green', 'GMM': 'red', 'Hierarchical': 'purple'}
    
    for method, row in method_stats.iterrows():
      plt.bar(method, row['mean'], yerr=row['std'], color=colors[method], capsize=5)
    
    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Average Silhouette Score")
    plt.title("Comparison of Clustering Algorithms")
    plt.show()
    
    
    
    # Presenting tables with means and standard deviations, followed by running ANOVA test and t-tests on the measurement results.
    
    # Calculate mean and standard deviation for each method
    method_stats = results_df.groupby('method')['silhouette'].agg(['mean', 'std'])
    method_stats
    
    
    
    kmeans_scores = results_df[results_df['method'] == 'KMeans']['silhouette']
    gmm_scores = results_df[results_df['method'] == 'GMM']['silhouette']
    hierarchical_scores = results_df[results_df['method'] == 'Hierarchical']['silhouette']
    
    # Perform ANOVA
    fvalue, pvalue = stats.f_oneway(kmeans_scores, gmm_scores, hierarchical_scores)
    print(f"ANOVA p-value: {pvalue}")
    
    # Perform pairwise t-tests
    def perform_t_test(group1, group2):
      t_statistic, p_value = stats.ttest_ind(group1, group2)
      return p_value
    
    # Create a dictionary to store p-values
    p_values = {
        'KMeans vs GMM': perform_t_test(kmeans_scores, gmm_scores),
        'KMeans vs Hierarchical': perform_t_test(kmeans_scores, hierarchical_scores),
        'GMM vs Hierarchical': perform_t_test(gmm_scores, hierarchical_scores)
    }
    
    # Create a DataFrame for p-values
    p_value_df = pd.DataFrame(list(p_values.items()), columns=['Comparison', 'p-value'])
    print("\nPairwise t-test p-values:")
    p_value_df



if __name__ == "__main__":
    run_algos_groups_dataframe()
    
