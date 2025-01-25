from flask import Flask, render_template, redirect
from dashboard import init_dashboard  # on importe la fonction
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

data_path = 'data/BankChurners.csv'
df = pd.read_csv(data_path)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')    
def dashboard():
    return redirect('/dash/')
    # Calcul de la répartition par genre

@app.route('/dashboard-front')
def front_dashboard():
    gender_counts = df['Gender'].value_counts().to_dict()

    # Conversion en listes
    gender_labels = list(gender_counts.keys())   # Ex.: ["F", "M"]
    gender_values = list(gender_counts.values()) # Ex.: [3, 3]

    return render_template(
        'dashboard.html',
        gender_labels=gender_labels,
        gender_values=gender_values
    )

if __name__ == '__main__':
    # On appelle init_dashboard(app) pour monter Dash sur /dash/
    init_dashboard(app)
    app.run(debug=True, port=8800)
