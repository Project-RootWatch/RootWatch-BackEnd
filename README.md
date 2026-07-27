# RootWatch — Backend

Flask REST API for the RootWatch smart soil, irrigation, and plant health
monitoring system.

## Role in the system

This is the hub that every other RootWatch component talks to:

- **RootWatch-Firmware** (ESP32-C6) sends sensor readings here over HTTP,
  and polls/receives irrigation trigger commands.
- **RootWatch-FrontEnd** (React) and **RootWatch-Mobile** (Flutter) both
  read from this same API — there is no client-specific backend logic,
  both clients hit identical endpoints.
- Owns the SQLite database of sensor history, hosts the trained LSTM
  forecasting model, and proxies requests to the Gemini API for Sinhala
  advisory text and plant photo analysis.

## Status

Early scaffold — see the main project plan for build order. Currently just
a health-check endpoint; sensor ingestion and storage land in the next step.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
```

Server runs at `http://localhost:5000`. Check `GET /health` to confirm it's up.

## Tech stack

- Flask (Python) — REST API
- SQLite — sensor reading storage
- TensorFlow/Keras — loads the pre-trained LSTM model (added later)
- Gemini API — advisory text + plant photo analysis
