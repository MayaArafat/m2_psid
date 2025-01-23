# Variables
VENV_DIR=venv

# Installation des dependances
install:
	@echo "[INFO] Installation des dependances..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "[INFO] Installation terminee."

# Lancement de l'application Flask
run:
	@echo "[INFO] Lancement de l'application Flask..."
	flask run --host=0.0.0.0 --port=5000

# Verification des dependances
check:
	pip freeze

# Nettoyage de l'environnement
clean:
	@echo "[INFO] Nettoyage des fichiers temporaires..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "[INFO] Nettoyage termine."

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  make install   - Installer les dependances"
	@echo "  make run       - Lancer l'application Flask"
	@echo "  make check     - Verifier les dependances installees"
	@echo "  make clean     - Nettoyer les fichiers temporaires"
	@echo "  make help      - Afficher cette aide"
