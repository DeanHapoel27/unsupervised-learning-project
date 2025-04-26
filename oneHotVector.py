from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

 
def run_onehotvector():
 categorical_features = ['class', 'gender', 'AgeCategory'] 
  # שמות המשתנים הקטגוריים
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
 def run_onehotvector():
   
