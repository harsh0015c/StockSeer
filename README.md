# StockSeer — Real CNN+LSTM Stock Price Prediction System

A full-stack stock prediction dashboard implementing the system described in:
> *"StockSeer: A Hybrid CNN-LSTM Framework for Real-Time Stock Price Prediction"*
> Divyanshi Rana, Harsh Chaudhary, Md. Emamoor Rasheed — IIMT College of Engineering, 2025-26

---

## What's Real in This Version

| Component | This repo |
|---|---|
| Deep learning model | ✅ Real TensorFlow/Keras CNN+LSTM |
| Training data | ✅ Live yfinance (Yahoo Finance) |
| Performance metrics | ✅ Computed on actual test-set (RMSE, MAE, MAPE, Directional Accuracy) |
| Technical indicators | ✅ RSI, Bollinger Bands, SMA, EMA — computed via NumPy |
| Auth | ✅ SHA-256 hashed passwords, Flask sessions |
| Stock comparison | ✅ Normalised 100-base charts |

---

## Model Architecture

```
Input (60 days × 1 feature)
  └── Conv1D(64, kernel=3, relu)
  └── Conv1D(64, kernel=3, relu)
  └── MaxPooling1D(2)
  └── LSTM(128, return_sequences=True)
  └── Dropout(0.2)
  └── LSTM(64)
  └── Dropout(0.2)
  └── Dense(32, relu)
  └── Dense(1)          ← predicted next closing price

Optimizer : Adam(lr=0.001)
Loss      : Huber
Callbacks : EarlyStopping(patience=10) + ReduceLROnPlateau(patience=5)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

**Default login:** `admin` / `admin123`

---

## How Training Works

The first time you request a prediction for a stock ticker:

1. The API returns `HTTP 202` with `"status": "training"`.
2. Training runs in a **background thread** (so the server stays responsive).
3. Poll `GET /api/stocks/<TICKER>/train_status` until `"status": "ready"`.
4. Then re-request the prediction — the model is loaded from disk (`.h5` file).

Subsequent requests for the same ticker load the cached model instantly.

---

## Project Structure

```
StockSeer/
├── app.py              ← Flask backend + real CNN+LSTM model
├── requirements.txt    ← Python dependencies (includes tensorflow)
├── users.json          ← Auto-created user database
├── models/             ← Saved .h5 model files (auto-created)
├── scalers/            ← Saved MinMaxScaler objects (auto-created)
└── static/
    └── index.html      ← Interactive dashboard frontend
```

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| POST | `/api/login` | Authenticate user |
| POST | `/api/register` | Create account |
| GET | `/api/stocks/list` | List supported tickers |
| GET | `/api/stocks/<T>/history?days=365` | OHLCV history (real yfinance) |
| GET | `/api/stocks/<T>/live` | Current price |
| GET | `/api/stocks/<T>/predict?days=30` | CNN+LSTM predictions |
| GET | `/api/stocks/<T>/train_status` | Check training progress |
| GET | `/api/stocks/<T>/indicators` | RSI, Bollinger, SMA, EMA |
| GET | `/api/stocks/compare?t1=AAPL&t2=MSFT` | Normalised comparison |
| GET | `/health` | System health check |

---

## Supported Tickers

US: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, NFLX  
India (NSE): TCS.NS, INFY, RELIANCE.NS, HDFCBANK.NS

---

## Deployment on Render

`render.yaml` is included. TensorFlow is included in `requirements.txt`.
Note: Render's free tier may time out during first-time model training for heavy tickers.
Use a paid plan or pre-train models locally and commit the `.h5` files.
