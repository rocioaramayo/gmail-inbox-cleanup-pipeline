# Gmail Inbox Cleanup

Proyecto para leer, clasificar y limpiar correos de Gmail desde Python.

## Qué incluye

- `outputs/gmail_ml_pipeline.py`: pipeline local sin IA externa, basado en `TF-IDF + LogisticRegression`
- `outputs/generate_gmail_token.py`: genera `token.json` para autenticar Gmail
- `outputs/gmail_gemini_cleanup_local.py`: versión anterior basada en Gemini
- `outputs/gmail_gemini_cleanup_colab.ipynb`: versión para Google Colab

## Requisitos

- Python 3.11+
- Gmail API habilitada en Google Cloud
- OAuth client de tipo `Desktop app`

## Instalación

```powershell
pip install scikit-learn joblib beautifulsoup4 google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Credenciales

Este repo **no incluye** credenciales reales.

Archivos locales esperados:

- `outputs/credentials.json`
- `outputs/token.json`

Esos archivos están ignorados por `.gitignore` y no deben subirse a GitHub.

Para generar `token.json`:

```powershell
cd .\outputs
python .\generate_gmail_token.py
```

## Uso del pipeline local

### 1. Pre-etiquetar correos

```powershell
cd .\outputs
python .\gmail_ml_pipeline.py prelabel --limit 100 --overwrite
```

### 2. Entrenar el modelo

```powershell
python .\gmail_ml_pipeline.py train
```

### 3. Probar clasificación

```powershell
python .\gmail_ml_pipeline.py predict --limit 50
```

Por defecto el pipeline usa:

```python
dry_run = True
```

Eso significa que **no borra correos reales**, solo simula la acción.

## Publicación segura

Antes de publicar:

1. Verificá que `outputs/credentials.json` y `outputs/token.json` no estén staged.
2. Verificá que no haya API keys o tokens copiados en notebooks, scripts o commits previos.
3. Si alguna credencial estuvo expuesta, **rotala** en Google Cloud / Google AI Studio antes de subir el repo.

## Notas

- El dataset y el modelo local también están ignorados por defecto porque pueden contener información personal del inbox.
- Si querés compartir un ejemplo de dataset, usá una copia anonimizada.
