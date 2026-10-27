# Image de l'API d'estimation (FastAPI). Construire : docker build -t prix-immobilier .
FROM python:3.13-slim

WORKDIR /app

# les dépendances d'abord : cette couche est réutilisée tant que le fichier ne change pas
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# uniquement ce qui sert à l'API : le code, le modèle et les données
COPY api.py .
COPY src/ src/
COPY models/ models/
COPY data/ data/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
