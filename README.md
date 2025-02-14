# m2_psid
## 1. Environnement 
    - Version python
    ```txt
    python == 3.9
    ```
    - Créer un environnement virtuel python :
    ```shell
    python3 -m venv venv

    venv/Scripts/activate
    venv/bin/activate # pour linux ou mac

    deactivate
    ```

## 2. Comandes pour lancer le jeu de données
make install
py app.py
## 2. Infos sur les données
### Données démographiques:
o	Customer_Age: âge du client.

o	Gender: sexe du client.

o	Dependent_count: nombre de personnes à charge.

o	Education_Level: niveau d’éducation.

o	Marital_Status: statut marital.

o	Income_Category: catégorie de revenus (ex.: “Moins de 40K”, “Plus de 120K”).
•	Données sur la relation client:
o	Months_on_book: ancienneté du client (en mois).

o	Total_Relationship_Count: nombre total de relations du client avec le fournisseur.

o	Contacts_Count_12_mon: nombre de contacts avec le client au cours des 12 derniers mois.

o	Card_Category: catégorie de carte (ex.: Blue, Silver, Gold).
### Données transactionnelles et financières:
o	Credit_Limit: limite de crédit accordée.

o	Total_Revolving_Bal: solde renouvelable total.

o	Avg_Open_To_Buy: ratio moyen de fonds disponibles pour les achats.

o	Total_Trans_Amt: montant total des transactions.

o	Total_Trans_Ct: nombre total de transactions.

o	Total_Amt_Chng_Q4_Q1: variation du montant total des transactions entre les trimestres Q4 et Q1.

o	Total_Ct_Chng_Q4_Q1: variation du nombre total de transactions entre Q4 et Q1.

o	Avg_Utilization_Ratio: taux moyen d’utilisation du crédit.
### Indicateur cible:
o	Attrition_Flag: indique si le client s’est désinscrit (“Attrited Customer”) ou est resté actif (“Existing Customer”).
Problématique générale
