import pandas as pd

def load_csv(file_path):
    """Load CSV data from the given file path."""
    return pd.read_csv(file_path)

def save_csv(data, file_path):
    """Save data to a CSV file."""
    data.to_csv(file_path, index=False)
