import matplotlib.pyplot as plt
from ml.loader import get_full_dataset
import seaborn as sns
import io
import base64
from sklearn.metrics import classification_report, confusion_matrix

def cechy(vol, sulfur, chlor, sulph):
    df = get_full_dataset()
    # Mapowanie kolorów HEX
    hex_palette = {0:'#d9dcd6', 1: "#8d0d3e"}

    # Wykresy
    plt.figure(figsize=(12, 5))

    # Wykres 1
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='volatile acidity', y='total sulfur dioxide',
                    hue='color_bin', palette=hex_palette, alpha=0.6, edgecolor=None)
    plt.scatter(vol, sulfur, color='black', marker='+', s=200, label='Twoje dane')
    plt.xlabel("volatile acidity")
    plt.ylabel("total sulfur dioxide")
    plt.title("Kwasowość lotna vs Całkowita zawartość dwutlenku siarki")
    plt.legend()

    # Wykres 2
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='chlorides', y='sulphates',
                    hue='color_bin', palette=hex_palette, alpha=0.6, edgecolor=None)
    plt.scatter(chlor, sulph, color='black', marker='+', s=200, label='Twoje dane')
    plt.xlabel("chlorides")
    plt.ylabel("sulphates")
    plt.title("Chlorki vs Siarczany")
    plt.legend()

    fig = plt.gcf()
    return convert_to_base64(fig)


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