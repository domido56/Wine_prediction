import joblib
from ml.loader import get_full_dataset, features, target
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_all_models():
    df = get_full_dataset()
    X = df[features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #X_test = scaler.fit_transform(X_test)

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

if __name__ == "__main__":
    train_all_models()