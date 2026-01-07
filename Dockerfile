FROM python:3.9-slim

# Empêcher python de buffer les logs
ENV PYTHONUNBUFFERED=1

# Dépendances système nécessaires (opencv, sam, torch)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Copier les fichiers
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

# Streamlit
EXPOSE 8501

# Commande de lancement
CMD ["streamlit", "run", "streamlit_sam_active_learning.py", "--server.port=8501", "--server.address=0.0.0.0"]
