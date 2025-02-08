import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Étape 1 : Importation des données
df = pd.read_csv("data/BankChurners.csv")

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

# Suppression des valeurs 'Unknown' dans les variables catégoriques
df = df[(df['Education_Level'] != 'Unknown') & (df['Income_Category'] != 'Unknown')]

# Regroupement des catégories de cartes
df['Card_Category'] = df['Card_Category'].replace(['Gold', 'Platinum', 'Silver'], 'Silver+')

# Encodage des variables catégoriques
label_enc = LabelEncoder()
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
for col in categorical_cols:
    df[col] = label_enc.fit_transform(df[col])

# Normalisation des variables numériques
num_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt', 
            'Total_Trans_Ct', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Étape 2 : Analyse en Composantes Principales (ACP)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[num_cols])
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Attrition_Flag'])
plt.title("Projection des clients selon PCA")
plt.show()

# Étape 3 : Clustering K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[num_cols])

# Étape 4 : Modélisation (Random Forest)
X = df.drop(columns=['Attrition_Flag'])
y = df['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_features= 0.8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
