import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import mutual_info_score




#k-means anomalies

print(len(X_scaled))

X_pca = PCA(n_components=2).fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_pca)

centers = kmeans.cluster_centers_
distances = np.linalg.norm(X_pca - centers[labels], axis=1)

mean_dist = np.mean(distances)
std_dist = np.std(distances)
threshold = mean_dist + 2.5 * std_dist

plt.figure(figsize=(10, 6))
plt.hist(distances, bins=30, edgecolor='black')
plt.axvline(x=threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.2f}")
plt.xlabel("Distance from Cluster Center")
plt.ylabel("Number of Points")
plt.title("Distribution of Distances from Cluster Centers")
plt.legend()
plt.tight_layout()
plt.show()

anomalies_k = distances > threshold
percentage = np.mean(anomalies_k) * 100
print(f"Detected {np.sum(anomalies_k)} anomalies out of {len(distances)} ({percentage:.2f}%)")


if not isinstance(X_scaled, pd.DataFrame):
    X_scaled_df = pd.DataFrame(X_scaled)
else:
    X_scaled_df = X_scaled.copy()

X_scaled_with_anomaly = X_scaled_df.copy()
X_scaled_with_anomaly['anomaly_kmeans'] = anomalies_k.astype(int)




#gmm anomalies

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

log_probs = gmm.score_samples(X_scaled)

mean_lp = np.mean(log_probs)
std_lp = np.std(log_probs)
threshold = mean_lp - 2.5 * std_lp

anomalies_gmm = log_probs < threshold

plt.figure(figsize=(10, 6))
plt.hist(log_probs, bins=30, color='skyblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
plt.xlabel('Log-Likelihood')
plt.ylabel('Frequency')
plt.title('Log-Likelihood Distribution with Anomaly Threshold (Mean - 3*STD)')
plt.legend()
plt.show()

print(f"Detected {np.sum(anomalies_gmm)} anomalies out of {len(X_scaled)} ({np.mean(anomalies)*100:.2f}%)")

X_scaled_with_anomaly['anomaly_gmm'] = anomalies_gmm.astype(int)



#isolation-forest anomalies

clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X_scaled)

scores = clf.decision_function(X_scaled)

mean_score = np.mean(scores)
std_score = np.std(scores)
threshold = mean_score - 2.5 * std_score

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=30, color='lightgreen', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
plt.xlabel('Isolation Forest Score')
plt.ylabel('Frequency')
plt.title('Anomaly Scores (Isolation Forest) with Threshold Line')
plt.legend()
plt.show()

anomalies_if = scores < threshold
print(f"Detected {np.sum(anomalies_if)} anomalies out of {len(X_scaled)} ({np.mean(anomalies_if)*100:.2f}%)")


X_scaled_with_anomaly['anomaly_iforest'] = anomalies_if.astype(int)



#one-class-svm anomalies

svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # nu = אחוז משוער של חריגים
svm.fit(X_scaled)


scores = svm.decision_function(X_scaled).ravel()  # ככל שקטן יותר → חריג יותר

mean_score = np.mean(scores)
std_score = np.std(scores)
threshold = mean_score - 2.5 * std_score


plt.figure(figsize=(10, 6))
plt.hist(scores, bins=30, color='lightblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
plt.xlabel('One-Class SVM Score')
plt.ylabel('Frequency')
plt.title('Anomaly Scores (One-Class SVM) with Threshold Line')
plt.legend()
plt.show()


anomalies_oc = scores < threshold
print(f"Detected {np.sum(anomalies_oc)} anomalies out of {len(X_scaled)} ({np.mean(anomalies_oc)*100:.2f}%)")

X_scaled_with_anomaly['anomaly_oneclass'] = anomalies_oc.astype(int)



#Mutual informatin between algorithms of anomalies detection to external variables




# Assuming X_scaled_with_anomaly, categorical_features, and new_df are defined

colors = {
    'anomaly_kmeans': 'blue',
    'anomaly_gmm': 'red',
    'anomaly_iforest': 'gray',
    'anomaly_oneclass': 'black'
}

categorical_cols = ['AgeCategory', 'class', 'gender']  # Replace with your actual categorical column names

algorithms = ['anomaly_gmm', 'anomaly_kmeans', 'anomaly_iforest', 'anomaly_oneclass']
mutual_info_values = {}

for algorithm in algorithms:
    mutual_info_values[algorithm] = {}
    for col in categorical_cols:
      mi = mutual_info_score(X_scaled_with_anomaly[algorithm], new_df[col])
      mutual_info_values[algorithm][col] = mi

# Plotting
plt.figure(figsize=(10, 6))
bar_width = 0.15
index = range(len(categorical_cols))

for i, algorithm in enumerate(algorithms):
  values = [mutual_info_values[algorithm][col] for col in categorical_cols]
  plt.bar([x + i * bar_width for x in index], values, bar_width, label=algorithm, color=colors[algorithm])

plt.xlabel("Categorical Features")
plt.ylabel("Mutual Information")
plt.title("Mutual Information between Anomaly Algorithms and Categorical Features")
plt.xticks([x + bar_width * (len(algorithms) -1) / 2 for x in index], categorical_cols)
plt.legend()
plt.tight_layout()
plt.show()
