import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    print("Dataset Shape:", data.shape)
    print("Missing values in each column:")
    print(data.isnull().sum())
    return data
