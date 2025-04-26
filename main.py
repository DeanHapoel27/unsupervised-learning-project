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


def main():
    run_getting_processing_data()
    run_onehotvector()
    run_clusters_dimensions_heatmaps()
    run_algos_groups_dataframe()
    run_statistical_tests()
    run_mutual_info_algos_ext_variables()
    run_anomalies_detection()
    run_tsne_class_k_means()

    






if __name__ == "__main__":
    main()
