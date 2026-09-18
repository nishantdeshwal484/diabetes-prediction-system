import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import pickle

# 1. Load the provided dataset
# Ensure 'diabetes.csv' is in the same directory
try:
    df = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please place it in the project folder.")
    exit()

# 2. Separate features and target variable
# Assuming standard Pima Indians Diabetes Dataset column names
X = df.drop('Outcome', axis=1) 
y = df['Outcome']

# 3. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create an ML Pipeline (Scaler + KNN)
# KNN requires feature scaling for optimal performance
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5)) 
])

# 5. Train the model
pipeline.fit(X_train, y_train)

# 6. Save the trained model as a .pkl file
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model trained successfully with KNN and saved as 'diabetes_model.pkl'")