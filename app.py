from flask import Flask, render_template
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
    # Calcul de la répartition par genre
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
    app.run(debug=True)
