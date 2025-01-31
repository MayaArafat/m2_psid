import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

df = pd.read_csv("data/BankChurners.csv")

def init_data() :
    # Forcer la colonne CLIENTNUM en string
    df['CLIENTNUM'] = df['CLIENTNUM'].astype(str)

    # Convertir les autres colonnes 'object' en 'category' (sauf CLIENTNUM)
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'CLIENTNUM':
            df[col] = df[col].astype('category')

    # Ajout d'une colonne d'utilisation du crédit (catégorisation)
    df['Utilization_Range'] = pd.cut(
        df['Avg_Utilization_Ratio'],
        bins=[0, 0.25, 0.50, 0.75, 1],
        labels=['Faible', 'Moyenne', 'Élevée', 'Très élevée']
    )
    return df

def encode_data(df):
    # DataFrame encodé (optionnel)
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')
    return df_encoded

# KPI
def getKpi(df) :
    nb_total_clients = len(df)
    taux_attrition = 0.0
    if 'Attrited Customer' in df['Attrition_Flag'].cat.categories:
        taux_attrition = (df['Attrition_Flag']
                            .value_counts(normalize=True)['Attrited Customer'] * 100).round(2)
    anciennete_moyenne = df['Months_on_book'].mean().round(1)

    return nb_total_clients, taux_attrition, anciennete_moyenne


# Repartition des genres (jauge)
def nbClients_par_genre(df):
    df_genre = df.groupby('Gender', observed=True).size().reset_index(name='Nombre de clients')
    fig = px.bar(
        df_genre,
        x='Gender',
        y='Nombre de clients',
        title="Répartition des clients par Genre",
        color='Gender',
        labels={
            'Gender': "Genre",
            'Nombre de clients': 'Nombre de clients'
        }
    )

    return fig


###############################################################################
# 3) CRÉATION DES VISUALISATIONS SANS FILTRE
###############################################################################

# A) Répartition des clients par statut d'attrition
def attrition_client(df) : 
    df_attrition = df.groupby('Attrition_Flag', observed=True).size().reset_index(name='Nombre de clients')
    fig_attrition = px.bar(
        df_attrition,
        x='Attrition_Flag',
        y='Nombre de clients',
        title="Répartition des clients par statut d'attrition",
        color='Attrition_Flag',
        labels={
            'Attrition_Flag': "Statut d'attrition",
            'Nombre de clients': 'Nombre de clients'
        }
    )
    fig_attrition.update_traces(
        hovertemplate="<b>Statut</b> : %{x}<br><b>Clients</b> : %{y}"
    )

    return fig_attrition

# B) Taux d'attrition (camembert)
def taux_attrition_chart(df, taux_attrition_value): 
    fig_attrition_gauge = px.pie(
        values=[taux_attrition_value, 100 - taux_attrition_value],
        names=['Clients désinscrits', 'Clients existants'],
        title="Taux d'attrition des clients",
        hole=0.4
    )
    fig_attrition_gauge.update_traces(
        hovertemplate="<b>%{label}</b> : %{percent}",
        textinfo='label+percent'
    )
    return fig_attrition_gauge


# E) Évolution du nombre moyen de transactions
def evol_nb_moyen_transactions(df) :
    df_grouped_transactions = df.groupby('Months_on_book', observed=True)['Total_Trans_Ct'].mean().reset_index()
    fig_transactions = px.line(
        df_grouped_transactions,
        x='Months_on_book',
        y='Total_Trans_Ct',
        title="Évolution du nombre moyen de transactions",
        labels={
            'Months_on_book': 'Ancienneté (mois)',
            'Total_Trans_Ct': 'Moyenne des transactions'
        }
    )
    return fig_transactions

    # F) Histogramme du taux d'utilisation
def hist_utilisation(df) :
    fig_util_hist = px.histogram(
        df,
        x='Avg_Utilization_Ratio',
        color='Attrition_Flag',
        title="Distribution du taux d'utilisation du crédit",
        barmode='overlay',
        nbins=20,
        labels={
            'Avg_Utilization_Ratio': 'Taux d\'utilisation du crédit',
            'Attrition_Flag': 'Statut d\'attrition'
        }
    )
    return fig_util_hist

# G) Montant moyen des transactions par catégorie de carte
def transac_moyen_by_cat(df) :
    df_card_mean = df.groupby('Card_Category', observed=True, as_index=False)['Total_Trans_Amt'].mean()
    df_card_mean.rename(columns={'Total_Trans_Amt': 'Montant moyen des transactions'}, inplace=True)
    fig_card_mean = px.bar(
        df_card_mean,
        x='Card_Category',
        y='Montant moyen des transactions',
        title="Montant moyen des transactions par catégorie de carte",
        labels={
            'Card_Category': 'Type de carte',
            'Montant moyen des transactions': 'Montant moyen'
        }
    )
    return fig_card_mean

# H) Sunburst (Revenu → Carte)
def sunburst_revenu_cart(df) :
    df_sun = df.groupby(['Income_Category', 'Card_Category'], observed=True).size().reset_index(name='Count')
    fig_sunburst = px.sunburst(
        df_sun,
        path=['Income_Category', 'Card_Category'],
        values='Count',
        title="Répartition hiérarchique Revenus → Carte",
        labels={
            'Income_Category': 'Revenus',
            'Card_Category': 'Carte'
        }
    )
    return fig_sunburst

# I) Parallel Categories
def get_parallel_categories(df) :
    df_parallel = df[['Attrition_Flag', 'Income_Category', 'Card_Category']].dropna()
    fig_parallel = px.parallel_categories(
        df_parallel,
        dimensions=['Attrition_Flag', 'Income_Category', 'Card_Category'],
        title="Analyse en parallèle (Attrition, Revenus, Carte)"
    )
    return fig_parallel

# Graoguqyes présent dans les détails 
# J) Répartition des clients par catégorie de revenues
def create_income_bar(df):
    df_income = df.groupby('Income_Category', observed=True).size().reset_index(name='Nombre de clients')
    fig = px.bar(
        df_income,
        x='Income_Category',
        y='Nombre de clients',
        title="Répartition des clients par catégorie de revenus",
        color='Income_Category',
        labels={
            'Income_Category': 'Revenus',
            'Nombre de clients': 'Nombre de clients'
        }
    )
    fig.update_traces(
        hovertemplate="<b>Revenus</b> : %{x}<br><b>Clients</b> : %{y}"
    )
    return fig
    
# I) Distribution de l'ancienneté 
def create_anciennete_histogram(df):
    fig = px.histogram(
        df,
        x='Months_on_book',
        color='Attrition_Flag',
        title="Distribution de l'ancienneté",
        barmode='overlay',
        nbins=20,
        labels={
            'Months_on_book': 'Ancienneté (mois)',
            'Attrition_Flag': "Statut d'attrition"
        }
    )
    fig.update_traces(
        opacity=0.6,
        hovertemplate="<b>Ancienneté</b> : %{x} mois<br><b>Clients</b> : %{y}"
    )
    return fig