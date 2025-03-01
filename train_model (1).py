import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to train a model for a given dataset
def train_model(dataset_path, target_column, model_filename):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

# Train models for each disease
train_model("diabetes.csv", "Outcome", "diabetes_model.pkl")
train_model("parkinsons.csv", "Status", "parkinsons_model.pkl")
train_model("lung_cancer.csv", "Lung_Cancer", "lung_cancer_model.pkl")
train_model("heart_disease.csv", "Heart_Disease", "heart_disease_model.pkl")
train_model("hypothyroidism.csv", "Thyroid", "hypothyroidism_model.pkl")

print("Models trained and saved successfully.")
