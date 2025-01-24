from flask import Flask, render_template, redirect
from dashboard import init_dashboard  # on importe la fonction
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return redirect('/dash/')

if __name__ == '__main__':
    # On appelle init_dashboard(app) pour monter Dash sur /dash/
    init_dashboard(app)
    app.run(debug=True, port=8800)
