from flask import Flask, render_template, redirect, jsonify
from dashboard import init_dashboard  # on importe la fonction
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.io as pio
from analytics_main import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')    
def dashboard():
    return redirect('/dash/')
    # Calcul de la répartition par genre

@app.route('/dashboard-front')
def front_dashboard():
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

    # Conversion en JSON pour transmission au template
    graph_fig_attrition = pio.to_json(fig_attrition)
    graph_taux_attrition = pio.to_json(fig_taux_attrition)
    graph_transactions = pio.to_json(fig_transactions, pretty=True)
    graph_hist_utilisation = pio.to_json(fig_hist_utilisation)
    graph_transac_moyen_by_cat = pio.to_json(fig_transac_moyen_by_cat)
    graph_sunburst_revenu_cart = pio.to_json(fig_sunburst_revenu_cart)
    graph_parallel_categories = pio.to_json(fig_parallel_categories)
    graph_nbclients_par_genre = pio.to_json(fig_nbclients_par_genre)


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
        graph_nbclients_par_genre_json = graph_nbclients_par_genre
    )



if __name__ == '__main__':
    # On appelle init_dashboard(app) pour monter Dash sur /dash/
    init_dashboard(app)
    app.run(debug=True, port=8800)
