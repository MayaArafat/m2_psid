import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, LeakyReLU, Dropout
from tensorflow.keras import Model

import pickle
import os

# Étape 1 : Importation des données
df = pd.read_csv("./Data/BankChurners.csv")

# Suppression des colonnes inutiles
drop_columns = ['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
df.drop(columns=drop_columns, inplace=True)

# Transformation des types des variables numériques
df['Credit_Limit'] = pd.to_numeric(df['Credit_Limit'], errors='coerce')
df['Avg_Open_To_Buy'] = pd.to_numeric(df['Avg_Open_To_Buy'], errors='coerce')
df['Total_Amt_Chng_Q4_Q1'] = pd.to_numeric(df['Total_Amt_Chng_Q4_Q1'], errors='coerce')
df['Total_Ct_Chng_Q4_Q1'] = pd.to_numeric(df['Total_Ct_Chng_Q4_Q1'], errors='coerce')
df['Avg_Utilization_Ratio'] = pd.to_numeric(df['Avg_Utilization_Ratio'], errors='coerce')

# Encodage des variables catégoriques
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Normalisation des variables numériques
num_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt', 
            'Total_Trans_Ct', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
            ]

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(columns=['Attrition_Flag'])
y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 2 : Analyse en Composantes Principales (ACP)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[num_cols])
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Attrition_Flag'])
plt.title("Projection des clients selon PCA")
plt.show()

# Étape 4 : Modélisation (Random Forest)
X = df.drop(columns=['Attrition_Flag'])
y = df['Attrition_Flag']
print(20*"*",X.columns, 20*"*")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_features= 0.8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("________ Random Forest ________")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Modèle ayant meilleur accuracy,matrice de confusion plus homogène, meilleurs résultats 
model = GradientBoostingClassifier(n_estimators=200, max_features= 0.8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("________ Gradient Boosting Classifier ________")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# =============================== SVM ===============================
# Modèle commenté car prend trop de temps lors des tests (pas utile de retester car il n'a pas de bons résultats)
"""

print("________ SVM Model (Linear)________")
# Train a SVM model
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

# Evaluate the model
y_pred = model_linear.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
"""
print("________ SVM Model (Sigmoid)________")
# Train a SVM model
model_linear = SVC(kernel='sigmoid')
model_linear.fit(X_train, y_train)

# Evaluate the model
y_pred = model_linear.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("________ SVM Model (Poly)________")
# Train a SVM model
model_linear = SVC(kernel='poly')
model_linear.fit(X_train, y_train)

# Evaluate the model
y_pred = model_linear.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("________ KNeighborsClassifier ________")

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("________ Nearest Centroid ________")
clf = NearestCentroid()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Model save and deployment

print("Model final : ", model)
# Créer un dossier pour stocker les modèles si ce n'est pas déjà fait
os.makedirs("ML/models", exist_ok=True)

with open("ML/models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("ML/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("ML/models/encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("ML/models/feature_order.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("ML/models/num_cols.pkl", "wb") as f:
    pickle.dump(num_cols, f)

# Sauvegarde des modèles PCA
with open("ML/models/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

print("Modèle, Scaler et Encodeurs sauvegardés avec succès ! ✅")
