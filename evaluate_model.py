import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the preprocessed test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
