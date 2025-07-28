import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

red = pd.read_csv('data/winequality-red.csv', sep=';')
white = pd.read_csv('data/winequality-white.csv', sep=';')

red['color_bin'] = 1
white['color_bin'] = 0

# polaczenie zbiorow
df = pd.concat([red, white], ignore_index=True)

# wybrane cechy na podstawie korelacji
features = ['volatile acidity', 'total sulfur dioxide', 'chlorides', 'sulphates']
X = df[features]
y = df['color_bin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)

svm = SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced')
svm.fit(X_train, y_train)

joblib.dump(rf, "models/rf.pkl")
joblib.dump(knn, "models/knn.pkl")
joblib.dump(svm, "models/svm.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Modele i scaler zapisane w folderze 'models'.")
