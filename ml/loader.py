import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = ['volatile acidity', 'total sulfur dioxide', 'chlorides', 'sulphates']
target = 'color_bin'

def get_full_dataset():
    red = pd.read_csv('data/winequality-red.csv', sep=';')
    white = pd.read_csv('data/winequality-white.csv', sep=';')

    red['color_bin'] = 1
    white['color_bin'] = 0

    # polaczenie zbiorow
    df = pd.concat([red, white], ignore_index=True)
    return df

def get_test_data():
    df = get_full_dataset()
    X = df[features]
    y = df[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    #SPRAWDZIC CZEMU TAK
    scaler = joblib.load("models/scaler.pkl")
    X_test = scaler.transform(X_test)

    return X_test, y_test

def load_models():
    rf = joblib.load("models/rf.pkl")
    knn = joblib.load("models/knn.pkl")
    svm = joblib.load("models/svm.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return rf, knn, svm, scaler

