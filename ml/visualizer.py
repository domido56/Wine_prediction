import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import classification_report, confusion_matrix

def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predykcja')
    ax.set_ylabel('Rzeczywista')
    ax.set_title('Macierz pomyłek')
    return convert_to_base64(fig)

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, ax=ax)
        ax.set_title('Ważność cech')
        return convert_to_base64(fig)
    return None

def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['Białe', 'Czerwone'])
    return report

def convert_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded