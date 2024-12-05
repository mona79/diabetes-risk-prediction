import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data_path = 'data/diabetes_generated.csv'
data = pd.read_csv(data_path)

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaled datasets and scaler
joblib.dump(scaler, 'models/scaler.pkl')
pd.DataFrame(X_train_scaled).to_csv('data/X_train.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False)

print("Data preprocessing complete. Scaled datasets and scaler saved.")
