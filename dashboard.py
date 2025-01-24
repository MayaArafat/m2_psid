from dash import Dash, dcc, html, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np

def init_dashboard(server):
    """
    Initialise l'application Dash sur le serveur Flask donné en paramètre.
    Les routes Dash seront accessibles via '/dash/'.
    Retourne l'objet dash_app (Dash).
    """

    ###############################################################################
    # 1) CRÉATION DE L'APPLICATION DASH
    ###############################################################################
    dash_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix='/dash/'
    )

    ###############################################################################
    # 2) CHARGEMENT ET PRÉPARATION DES DONNÉES
    ###############################################################################
    df = pd.read_csv("data/BankChurners.csv")

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

    # DataFrame encodé (optionnel)
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')

    # KPI
    nb_total_clients = len(df)
    taux_attrition = 0.0
    if 'Attrited Customer' in df['Attrition_Flag'].cat.categories:
        taux_attrition = (df['Attrition_Flag']
                          .value_counts(normalize=True)['Attrited Customer'] * 100).round(2)
    anciennete_moyenne = df['Months_on_book'].mean().round(1)

    ###############################################################################
    # 3) CRÉATION DES VISUALISATIONS SANS FILTRE
    ###############################################################################

    # A) Répartition des clients par statut d'attrition
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

    # B) Taux d'attrition (camembert)
    fig_attrition_gauge = px.pie(
        values=[taux_attrition, 100 - taux_attrition],
        names=['Clients désinscrits', 'Clients existants'],
        title="Taux d'attrition des clients",
        hole=0.4
    )
    fig_attrition_gauge.update_traces(
        hovertemplate="<b>%{label}</b> : %{percent}",
        textinfo='label+percent'
    )
         # E) Évolution du nombre moyen de transactions
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

    # F) Histogramme du taux d'utilisation
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

    # G) Montant moyen des transactions par catégorie de carte
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

    # H) Sunburst (Revenu → Carte)
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

    # I) Parallel Categories
    df_parallel = df[['Attrition_Flag', 'Income_Category', 'Card_Category']].dropna()
    fig_parallel = px.parallel_categories(
        df_parallel,
        dimensions=['Attrition_Flag', 'Income_Category', 'Card_Category'],
        title="Analyse en parallèle (Attrition, Revenus, Carte)"
    )


    # Fonctions dynamiques (pour filtres)
    def create_anciennete_histogram(df_sub):
        fig = px.histogram(
            df_sub,
            x='Months_on_book',
            color='Attrition_Flag',
            title="Distribution de l'ancienneté (filtrée)",
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

    def create_income_bar(df_sub):
        df_income = df_sub.groupby('Income_Category', observed=True).size().reset_index(name='Nombre de clients')
        fig = px.bar(
            df_income,
            x='Income_Category',
            y='Nombre de clients',
            title="Répartition des clients par catégorie de revenus (filtrée)",
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

    ###############################################################################
    # 4) ONGLET DONNÉES DÉTAILLÉES (Recherche + Table)
    ###############################################################################
    client_info_layout = html.Div(id='client-info', style={
        'backgroundColor': '#F8F9F9',
        'padding': '10px',
        'marginBottom': '20px',
        'border': '1px solid #D6DBDF',
        'borderRadius': '5px'
    })

    search_layout = html.Div([
        html.Label("Rechercher un client via CLIENTNUM :"),
        dcc.Input(id='clientnum-input', type='text', placeholder='Entrez un CLIENTNUM', style={'marginLeft': '10px'}),
        html.Button("Rechercher", id='btn-search', n_clicks=0, style={'marginLeft': '10px'}),
    ], style={'marginBottom': '20px'})

    table_layout = html.Div([
        html.P("Vous pouvez filtrer et trier les enregistrements dans le tableau ci-dessous."),
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '120px', 'maxWidth': '250px'}
        )
    ])

    detailed_layout = html.Div([
        search_layout,
        client_info_layout,
        table_layout
    ])

    ###############################################################################
    # 5) ONGLET REVENUS & ANCIENNETÉ : FILTRES INTERACTIFS
    ###############################################################################
    filters_layout = html.Div([
        html.Label("Choisissez une catégorie de revenu pour filtrer :", style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='income-filter',
            options=[{'label': cat, 'value': cat} for cat in df['Income_Category'].cat.categories]
                    + [{'label': 'Toutes', 'value': 'Toutes'}],
            value='Toutes',
            clearable=False,
            style={'width': '300px'}
        ),
        html.Div(id='filtered-charts', children=[
            dcc.Graph(id='fig-income-filtered'),
            dcc.Graph(id='fig-anciennete-filtered')
        ])
    ])

    ###############################################################################
    # 6) LAYOUT GLOBAL
    ###############################################################################
    dash_app.layout = html.Div(children=[
        html.H1(
            "Analyse des Clients Bancaires",
            style={'textAlign': 'center', 'color': '#003366'}
        ),
        html.Div([
            html.H3(f"Nombre total de clients : {nb_total_clients}"),
            html.H3(f"Taux d'attrition : {taux_attrition}%"),
            html.H3(f"Ancienneté moyenne des clients : {anciennete_moyenne} mois")
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Tabs([
            # Onglet Aperçu Général
            dcc.Tab(label='Aperçu Général', children=[
                dcc.Graph(figure=fig_attrition),
                dcc.Graph(figure=fig_attrition_gauge)
            ]),

            # Onglet Revenus et Ancienneté => filtres
            dcc.Tab(label='Revenus et Ancienneté', children=[
                html.Div(
                    "Explorez la répartition par revenus et l'ancienneté des clients, "
                    "avec filtre sur la catégorie de revenus :",
                    style={'marginBottom': '10px', 'fontStyle': 'italic'}
                ),
                filters_layout
            ]),

            # Onglet Transactions
            dcc.Tab(label='Transactions', children=[
            dcc.Graph(figure=fig_transactions),
            dcc.Graph(figure=fig_util_hist),
            dcc.Graph(figure=fig_card_mean)            ]),

            # Onglet Analyses Multi-Dimensions
            dcc.Tab(label='Analyses Multi-Dimensions', children=[
            dcc.Graph(figure=fig_sunburst),
            dcc.Graph(figure=fig_parallel)            ]),

            # Onglet Données Détaillées
            dcc.Tab(label='Données Détaillées', children=[
                detailed_layout
            ])
        ]),
    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#FAFAFA', 'margin': '20px'})

    ###############################################################################
    # 7) CALLBACK POUR LES FILTRES (Income_Category)
    ###############################################################################
    @dash_app.callback(
        [Output('fig-income-filtered', 'figure'),
         Output('fig-anciennete-filtered', 'figure')],
        [Input('income-filter', 'value')]
    )
    def update_income_anciennete(income_choice):
        """
        Si income_choice != 'Toutes', on filtre le DataFrame sur cette catégorie de revenus.
        Puis on génère 2 figures :
           - create_income_bar(...)
           - create_anciennete_histogram(...)
        """
        if income_choice == 'Toutes':
            df_filtered = df
        else:
            df_filtered = df[df['Income_Category'] == income_choice]

        fig_income_filtered = create_income_bar(df_filtered)
        fig_anciennete_filtered = create_anciennete_histogram(df_filtered)
        return fig_income_filtered, fig_anciennete_filtered

    ###############################################################################
    # 8) CALLBACK RECHERCHE CLIENT
    ###############################################################################
    @dash_app.callback(
        Output('client-info', 'children'),
        [Input('btn-search', 'n_clicks')],
        [State('clientnum-input', 'value')]
    )
    def search_client(n_clicks, client_id):
        if n_clicks > 0 and client_id:
            match = df.loc[df['CLIENTNUM'] == client_id.strip()]
            if len(match) == 1:
                row = match.iloc[0]
                return html.Div([
                    html.H4("Informations sur le client :"),
                    html.P(f"CLIENTNUM : {row['CLIENTNUM']}"),
                    html.P(f"Statut d'attrition : {row['Attrition_Flag']}"),
                    html.P(f"Genre : {row['Gender']}"),
                    html.P(f"Âge du client : {row['Customer_Age']} ans"),
                    html.P(f"Catégorie de revenu : {row['Income_Category']}"),
                    html.P(f"Catégorie de carte : {row['Card_Category']}"),
                    html.P(f"Ancienneté (mois) : {row['Months_on_book']}"),
                    html.P(f"Utilization Range : {row['Utilization_Range']}"),
                ])
            elif len(match) > 1:
                return html.Div("Plusieurs correspondances trouvées !", style={'color': 'red'})
            else:
                return html.Div("Aucun client trouvé avec ce CLIENTNUM.", style={'color': 'red'})
        return "Saisissez un CLIENTNUM puis cliquez sur Rechercher."

    ###############################################################################
    # 9) RETOUR DE L'OBJET dash_app
    ###############################################################################
    return dash_app
