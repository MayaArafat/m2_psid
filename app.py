from flask import Flask, render_template, request, jsonify
from dashboard import init_dashboard
import pandas as pd
import plotly.io as pio
import pickle
import os

# Fonction pour charger un fichier pickle en toute sécurité
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path}: {e}")
        return None

# Chargement des modèles et transformateurs
model = load_pickle("ML/models/model.pkl")
scaler = load_pickle("ML/models/scaler.pkl")
label_encoders = load_pickle("ML/models/encoders.pkl")
feature_order = load_pickle("ML/models/feature_order.pkl")
num_cols = load_pickle("ML/models/num_cols.pkl")
pca = load_pickle("ML/models/pca.pkl")
kmeans = load_pickle("ML/models/kmeans.pkl")

print("Modèles et transformateurs chargés avec succès ! ✅")

app = Flask(__name__)
# On appelle init_dashboard(app) pour monter Dash sur /dash/
dash_app = init_dashboard(app)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data_analytics')    
def data_analytics():
    return render_template("data_analytics.html", dash_app_placeholder=dash_app.index())


@app.route('/dashboard')
def dashboard():
    df = init_data()    
    nb_total_clients, taux_attrition_value, anciennete_moyenne = getKpi(df)
    
    # Génération des graphiques
    fig_attrition = attrition_client(df)
    fig_taux_attrition = taux_attrition_chart(df, taux_attrition_value)
    fig_transactions = evol_nb_moyen_transactions(df)
    fig_hist_utilisation = hist_utilisation(df)
    fig_transac_moyen_by_cat = transac_moyen_by_cat(df)
    fig_sunburst_revenu_cart = sunburst_revenu_cart(df)
    fig_parallel_categories = get_parallel_categories(df)
    fig_nbclients_par_genre = nbClients_par_genre(df)
    fig_client_par_revenu = create_income_bar(df)
    fig_anciennete = create_anciennete_histogram(df)

    # Conversion en JSON pour transmission au template
    graph_fig_attrition = pio.to_json(fig_attrition)
    graph_taux_attrition = pio.to_json(fig_taux_attrition)
    graph_transactions = pio.to_json(fig_transactions, pretty=True)
    graph_hist_utilisation = pio.to_json(fig_hist_utilisation)
    graph_transac_moyen_by_cat = pio.to_json(fig_transac_moyen_by_cat)
    graph_sunburst_revenu_cart = pio.to_json(fig_sunburst_revenu_cart)
    graph_parallel_categories = pio.to_json(fig_parallel_categories)
    graph_nbclients_par_genre = pio.to_json(fig_nbclients_par_genre)
    graph_client_par_revenu = pio.to_json(fig_client_par_revenu)
    graph_anciennete = pio.to_json(fig_anciennete)


    return render_template(
        'dashboard.html',
        kpi1=nb_total_clients,
        kpi2=taux_attrition_value,
        kpi3=anciennete_moyenne,
        fig_attrition_json=graph_fig_attrition,
        taux_attrition_json=graph_taux_attrition,
        transactions_json=graph_transactions,
        hist_utilisation_json = graph_hist_utilisation,
        transac_moyen_by_cat_json = graph_transac_moyen_by_cat,
        sunburst_revenu_cart_json = graph_sunburst_revenu_cart,
        parallel_categories_json = graph_parallel_categories,
        graph_nbclients_par_genre_json = graph_nbclients_par_genre,
        graph_client_par_revenu_json = graph_client_par_revenu,
        graph_anciennete_json = graph_anciennete
    )

# Formulaire pour utiliser le model
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', prediction=None)  # Afficher la page vide

    data = {col: request.form.get(col) for col in num_cols}
    for col in num_cols:
        if data[col] is None:
            return f"Erreur : {col} est manquant ou vide", 400
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Encodage des variables catégoriques
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    for col in categorical_cols:
       
        if col in request.form and col in label_encoders:
            print(col)
            data[col] = label_encoders[col].transform([request.form[col]])[0]
            # if col == "Education_Level" :
            #     print("Education level ", data[col])
        else:
            data[col] = 0  # Valeur par défaut pour éviter les erreurs

    # Création du DataFrame
    df_input = pd.DataFrame([data])
    print("Avant transformation :", df_input.columns.tolist())

    # Séparer les colonnes numériques et catégoriques
    df_input_scaled = df_input.copy()
    num_cols_only = [col for col in num_cols if col not in categorical_cols]
    df_input_scaled[num_cols_only] = scaler.transform(df_input[num_cols_only])
    print("Après transformation :", df_input_scaled.columns.tolist())
    print(f"---Taille : {df_input_scaled.shape}")


    # Appliquer la transformation PCA
    pca_result = pca.transform(df_input_scaled[num_cols])
    df_input_scaled['PCA1'], df_input_scaled['PCA2'] = pca_result[:, 0], pca_result[:, 1]
    df_input_scaled['Cluster'] = kmeans.predict(df_input[num_cols])

    print("PCA1: ", df_input_scaled['PCA1'])
    print("PCA2: ", df_input_scaled['PCA2'])

    for col in feature_order:
        if col not in df_input_scaled:
            df_input_scaled[col] = 0
    df_input_scaled = df_input_scaled[feature_order]

    print("Colonnes envoyées au modèle :", df_input_scaled.columns.tolist())
    print(20*"*", "Modèle :", model)
    prediction = model.predict(df_input_scaled)[0]
    print(prediction)
    # Conversion du résultat
    result = "Le client est actif" if prediction == 'Existing Customer' else "Le client risque de se désinscrire"
    print("Prédiction : ", result)
    return render_template('predict.html', prediction=result)



if __name__ == '__main__':
    app.run(debug=True, port=8800)
