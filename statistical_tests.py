import pandas as pd
from scipy import stats

#T-test for each pair among the 3 better algorithms, ANOVA test to check if there is a significant difference between the three algorithms.


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

#T-test between the Birch algorithm and each of the other algorithms.

birch_scores = results_df[results_df['method'] == 'Birch']['silhouette']
kmeans_scores = results_df[results_df['method'] == 'KMeans']['silhouette']
gmm_scores = results_df[results_df['method'] == 'GMM']['silhouette']
hierarchical_scores = results_df[results_df['method'] == 'Hierarchical']['silhouette']

# Perform pairwise t-tests
def perform_t_test(group1, group2):
  t_statistic, p_value = stats.ttest_ind(group1, group2)
  return p_value

# Create a dictionary to store p-values
p_values = {
    'Birch vs KMeans': perform_t_test(birch_scores, kmeans_scores),
    'Birch vs GMM': perform_t_test(birch_scores, gmm_scores),
    'Birch vs Hierarchical': perform_t_test(birch_scores, hierarchical_scores),
}

# Create a DataFrame for p-values
p_value_df = pd.DataFrame(list(p_values.items()), columns=['Comparison', 'p-value'])
print("\nPairwise t-test p-values:")
p_value_df
