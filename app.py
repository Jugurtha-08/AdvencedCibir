import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import os
from distances import manhattan, euclidean, chebyshev, canberra
import streamlit as st
from PIL import Image
from descriptors import glcm, bitdesc, haralick_feat, bit_glcm_haralick

# Configuration du thème
st.set_page_config(
    page_title="CBIR App",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Charger les signatures
signatures_glcm = np.load('signatures_glcm.npy', allow_pickle=True)
signatures_bitdesc = np.load('signatures_bitdesc.npy', allow_pickle=True)
signatures_haralick = np.load('signatures_haralick_feat.npy', allow_pickle=True)
signatures_combined = np.load('signatures_bit_glcm_haralick.npy', allow_pickle=True)

# Définir les fonctions de descripteurs
descriptor_functions = {
    'GLCM': glcm,
    'BIT': bitdesc,
    'Haralick': haralick_feat,
    'BiT_Glcm_haralick': bit_glcm_haralick
}

# Définir les fonctions de distance
distance_functions = {
    'Manhattan': manhattan,
    'Euclidean': euclidean,
    'Chebyshev': chebyshev,
    'Canberra': canberra
}

# Fonction d'extraction des caractéristiques
def extract_features(image, descriptor_choice):
    descriptor_func = descriptor_functions[descriptor_choice]
    return descriptor_func(image)

# Affiner les modèles avec GridSearchCV
def fine_tune_model(model, param_grid, X_train, Y_train):
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    return grid_search.best_estimator_

def fine_tune_lda(X_train, Y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)  # Entraînez le modèle ici
    return model  # Pas d'hyperparamètres à ajuster pour LDA

def fine_tune_knn(X_train, Y_train):
    param_grid = {'n_neighbors': list(range(1, 30)), 'p': [1, 2]}  # Ajout de 'p' pour la métrique de distance
    return fine_tune_model(KNeighborsClassifier(), param_grid, X_train, Y_train)

def fine_tune_svm(X_train, Y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
    return fine_tune_model(SVC(), param_grid, X_train, Y_train)

def fine_tune_random_forest(X_train, Y_train):
    param_grid = {'n_estimators': [10, 50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [None, 10, 20, 30]}
    return fine_tune_model(RandomForestClassifier(), param_grid, X_train, Y_train)

def fine_tune_adaboost(X_train, Y_train):
    param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    return fine_tune_model(AdaBoostClassifier(), param_grid, X_train, Y_train)

def fine_tune_decision_tree(X_train, Y_train):
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    return fine_tune_model(DecisionTreeClassifier(), param_grid, X_train, Y_train)

def fine_tune_naive_bayes(X_train, Y_train):
    return GaussianNB()

# Application des styles personnalisés
st.markdown("""
    <style>
        .stApp {
            background-color: #F5F5F5; /* Couleur de fond clair */
            color: #333333; /* Texte sombre pour le contraste */
        }
        .css-1d391kg, .css-1v3fvcr {
            background-color: #FFFFFF; /* Fond des widgets */
            border-radius: 8px; /* Coins arrondis pour un look moderne */
            padding: 10px; /* Espacement interne */
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Ombre légère pour un effet de profondeur */
        }
        .css-1aumxhk {
            color: #1E88E5; /* Couleur des titres */
        }
        .stButton, .stFileUploader, .stSelectbox, .stRadio {
            border-radius: 8px; /* Coins arrondis pour les boutons et sélecteurs */
            border: 1px solid #DDDDDD; /* Bordure légère */
            padding: 10px; /* Espacement interne */
        }
        .stButton:hover, .stFileUploader:hover, .stSelectbox:hover, .stRadio:hover {
            background-color: #E3F2FD; /* Couleur de survol pour les boutons et sélecteurs */
        }
        .center-upload {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .center-upload > div {
            width: 50%;
        }
        .stTitle {
            font-size: 24px; /* Taille des titres */
            font-weight: bold; /* Poids de la police */
        }
        .stHeader {
            font-size: 20px; /* Taille des en-têtes */
            font-weight: bold; /* Poids de la police */
        }
    </style>
    """, unsafe_allow_html=True)


def cbir_basic():
    st.sidebar.header("Descriptor")
    descriptor_choice = st.sidebar.selectbox("Choose Descriptor", ["GLCM", "BIT", "Haralick", "BiT_Glcm_haralick"])
    st.sidebar.header("Distances")
    distance_choice = st.sidebar.selectbox("Choose Distance Metric", ["Manhattan", "Euclidean", "Chebyshev", "Canberra"])
    st.sidebar.header("Number of Images")
    image_count = st.sidebar.number_input("Number of Images", min_value=1, value=4, step=1)
    st.title("Jugurtha Content-based Image Retrieval")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        uploaded_image_features = extract_features(img, descriptor_choice)

        # Select the correct signatures based on the descriptor
        if descriptor_choice == 'GLCM':
            signatures = signatures_glcm
        elif descriptor_choice == 'BIT':
            signatures = signatures_bitdesc
        else:
            signatures = signatures_combined

        distances = []
        dist_func = distance_functions[distance_choice]

        for signature in signatures:
            feature_vector = np.array(signature[:-3], dtype=float)  # Exclude the last three elements (relative path, folder name, class label)
            dist = dist_func(uploaded_image_features, feature_vector)
            distances.append((dist, signature[-3], signature[-2], signature[-1]))  # Keep the relative path, folder name, and class label

        distances.sort(key=lambda x: x[0])

        st.header(f"Top {image_count} Similar Images")
        cols = st.columns(4)
        for i in range(image_count):
            if i >= len(cols):  
                break
            dist, relative_path, folder_name, class_label = distances[i]
            img_path = os.path.join('images', relative_path)
            similar_img = Image.open(img_path)
            cols[i % 4].image(similar_img, caption=f"{folder_name}", use_column_width=True)
    else:
        st.write("Please upload an image to start.")

def cbirAdvanced():
    st.sidebar.header("Descriptor")
    descriptor_choice = st.sidebar.selectbox("Choose Descriptor", ["GLCM", "BIT", "Haralick", "BiT_Glcm_haralick"])
    st.sidebar.header("Distances")
    distance_choice = st.sidebar.selectbox("Choose Distance Metric", ["Manhattan", "Euclidean", "Chebyshev", "Canberra"])
    st.sidebar.header("Number of Images")
    image_count = st.sidebar.number_input("Number of Images", min_value=1, value=4, step=1)
    st.write("Select Classification Algorithm")
    classifier = st.selectbox("Classifier", ["LDA", "KNN", "Naive Bayes", "Decision Tree", "SVM", "Random Forest", "AdaBoost"])
    
    st.title("Jugurtha Content-based Image Retrieval (Advanced)")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        uploaded_image_features = extract_features(img, descriptor_choice)

        if descriptor_choice == 'GLCM':
            signatures = signatures_glcm
        elif descriptor_choice == 'BIT':
            signatures = signatures_bitdesc
        else:
            signatures = signatures_combined

        X = np.array([np.array(sig[:-3], dtype=float) for sig in signatures])
        Y = np.array([sig[-1] for sig in signatures])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        if classifier == "LDA":
            model = fine_tune_lda(X_train, Y_train)
        elif classifier == "KNN":
            model = fine_tune_knn(X_train, Y_train)
        elif classifier == "SVM":
            model = fine_tune_svm(X_train, Y_train)
        elif classifier == "Random Forest":
            model = fine_tune_random_forest(X_train, Y_train)
        elif classifier == "AdaBoost":
            model = fine_tune_adaboost(X_train, Y_train)
        elif classifier == "Decision Tree":
            model = fine_tune_decision_tree(X_train, Y_train)
        elif classifier == "Naive Bayes":
            model = fine_tune_naive_bayes(X_train, Y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_accuracy = accuracy_score(Y_train, train_predictions)
        test_accuracy = accuracy_score(Y_test, test_predictions)
        train_f1 = f1_score(Y_train, train_predictions, average='weighted')
        test_f1 = f1_score(Y_test, test_predictions, average='weighted')
        train_precision = precision_score(Y_train, train_predictions, average='weighted')
        test_precision = precision_score(Y_test, test_predictions, average='weighted')
        train_recall = recall_score(Y_train, train_predictions, average='weighted')
        test_recall = recall_score(Y_test, test_predictions, average='weighted')

        st.write(f"Training Accuracy: {train_accuracy:.2f}")
        st.write(f"Test Accuracy: {test_accuracy:.2f}")
        st.write(f"Training F1 Score: {train_f1:.2f}")
        st.write(f"Test F1 Score: {test_f1:.2f}")
        st.write(f"Training Precision: {train_precision:.2f}")
        st.write(f"Test Precision: {test_precision:.2f}")
        st.write(f"Training Recall: {train_recall:.2f}")
        st.write(f"Test Recall: {test_recall:.2f}")

        # Compute distances between the uploaded image and all the signatures
        distances = []
        dist_func = distance_functions[distance_choice]

        for signature in signatures:
            feature_vector = np.array(signature[:-3], dtype=float)  # Exclude the last three elements (relative path, folder name, class label)
            dist = dist_func(uploaded_image_features, feature_vector)
            distances.append((dist, signature[-3], signature[-2], signature[-1]))  # Keep the relative path, folder name, and class label

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[0])

        # Display the top N similar images
        st.header(f"Top {image_count} Similar Images")
        cols = st.columns(4)
        for i in range(image_count):
            if i >= len(cols):  # Check to avoid index errors
                break
            dist, relative_path, folder_name, class_label = distances[i]
            img_path = os.path.join('images', relative_path)
            similar_img = Image.open(img_path)
            cols[i % 4].image(similar_img, caption=f"{folder_name}", use_column_width=True)
    else:
        st.write("Please upload an image to start.")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Basic CBIR", "Advanced CBIR"])
    
    if app_mode == "Basic CBIR":
        cbir_basic()
    else:
        cbirAdvanced()

if __name__ == "__main__":
    main()
