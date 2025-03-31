# My Trading System

Sistema di trading automatizzato basato su Machine Learning (LightGBM), con supporto a:
- Preprocessing e feature engineering su dati OHLCV
- Addestramento del modello con Optuna per ottimizzazione automatica degli iperparametri
- Inference e generazione segnali
- Simulazione di paper trading con gestione TP/SL/timeout e stato persistente

## Struttura del progetto

- `feature_engineering.py` – Calcolo degli indicatori tecnici e feature
- `tune_model_optuna.py` – Addestramento e tuning del modello
- `inference.py` – Inference del modello su dati futuri
- `paper_trading.py` – Simulazione paper trading
- `utils/` – Funzioni di utilità condivise
- `data/` – CSV e file locali per le run di test

## Requisiti

- Python 3.10+
- Librerie:
  - `pandas`, `numpy`, `lightgbm`
  - `optuna`, `scikit-learn`, `ta`
  - `psycopg2` (per connessione al DB PostgreSQL)

Installa i requisiti:

```bash
pip install -r requirements.txt
```

## Esecuzione

Esempio di training:

```bash
python tune_model_optuna.py
```

Esempio di paper trading:

```bash
python start_session.py
```

## Database

Il sistema salva metriche di training nel database PostgreSQL (locale o AWS). Le run di inference e trading sono salvate come CSV in locale.

## Privacy & Sicurezza

Nessun dato viene trasmesso verso terze parti. L’intero sistema è progettato per girare in locale e rispettare la privacy dell’utente.


