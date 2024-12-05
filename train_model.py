import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the preprocessed data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/random_forest_model.pkl')

print("Model training complete. Model saved at 'models/random_forest_model.pkl'.")
