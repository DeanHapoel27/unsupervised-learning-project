from scipy.spatial import ConvexHull

def run_tsne_class_k_means():

    # Assuming X_scaled and new_df are defined from the previous code block
    
    # Apply PCA
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # KMeans clustering on t-SNE results
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_tsne)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Original class labels
    unique_labels = np.unique(new_df['class'])
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    for label in unique_labels:
        indices = new_df['class'] == label
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=label, color=colors(np.where(unique_labels==label)[0][0]), alpha=0.7)
    
    # Plot cluster boundaries with convex hulls
    for cluster_label in range(4):
        cluster_indices = kmeans_labels == cluster_label
        cluster_points = X_tsne[cluster_indices]
        hull = ConvexHull(cluster_points)
        plt.plot(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], 'k-', linewidth=1)
    
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization of Clusters with Original Class Labels (Full Dataset)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_tsne_class_k_means()
    
