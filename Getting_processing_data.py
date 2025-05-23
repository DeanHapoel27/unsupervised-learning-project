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


def run_getting_processing_data():
  df = pd.read_csv('bodyPerformance.csv')
  df.head()
  
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
  
  \\one-hot-vector
  
  categorical_features = ['class', 'gender', 'AgeCategory'] 
  
  numeric_features = ['height_cm','weight_kg','body fat_%','diastolic',
   'systolic','gripForce','sit and bend forward_cm','sit-ups counts','broad jump_cm',]  # שמות הנומריים
  
  preprocessor = ColumnTransformer(
      transformers=[
          ('cat', OneHotEncoder(), categorical_features)
      ],
      remainder='passthrough'
  )
  X_encoded = preprocessor.fit_transform(new_df)
  
  
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_encoded)

if __name__ == "__main__":
  run_getting_processing_data()
    
