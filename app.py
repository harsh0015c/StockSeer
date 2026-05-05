
from flask import Flask, request, jsonify, send_from_directory, session
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, hashlib, os, threading, time, warnings

warnings.filterwarnings("ignore")

# ── Optional TensorFlow import (graceful fallback for cold environments) ──────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not installed — install it with: pip install tensorflow>=2.13.0")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf

app = Flask(__name__, static_folder='static')
app.secret_key = 'stockseer_real_2025'

os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ── USER DATABASE ─────────────────────────────────────────────────────────────
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    default = {"admin": {"password": hash_password("admin123"),
                         "name": "Admin User",
                         "created": str(datetime.now())}}
    save_users(default)
    return default

def save_users(u):
    with open(USERS_FILE, 'w') as f:
        json.dump(u, f, indent=2)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()


# ── REAL CNN+LSTM PREDICTOR ───────────────────────────────────────────────────
SEQ_LEN = 60          # lookback window (days)
TRAIN_RATIO = 0.80    # 80/20 split as stated in report

# Thread-safe training status store
_train_status = {}     # ticker -> {"status": "training"|"ready"|"error", "metrics": {...}}
_train_lock   = threading.Lock()


def _build_cnnlstm(seq_len: int, n_features: int = 1):

    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(seq_len, n_features)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber()
    )
    return model


def _make_sequences(scaled_prices: np.ndarray, seq_len: int):
    """Turn a 1-D scaled price array into (X, y) supervised pairs."""
    X, y = [], []
    for i in range(seq_len, len(scaled_prices)):
        X.append(scaled_prices[i - seq_len:i])
        y.append(scaled_prices[i, 0])
    return np.array(X), np.array(y)


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, Directional Accuracy — all real values."""
    mae  = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    # MAPE — avoid div-by-zero
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    # Directional accuracy
    actual_dir = np.sign(np.diff(actual))
    pred_dir   = np.sign(np.diff(predicted))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)
    return {
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "directional_accuracy": round(dir_acc, 1)
    }


def _train_and_cache(ticker: str, prices: list):

    with _train_lock:
        _train_status[ticker] = {"status": "training", "metrics": None}

    try:
        arr = np.array(prices, dtype=float).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(arr)

        X, y = _make_sequences(scaled, SEQ_LEN)
        split = int(len(X) * TRAIN_RATIO)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if TF_AVAILABLE:
            model = _build_cnnlstm(SEQ_LEN, n_features=1)
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
            ]
            model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=0
            )
            model.save(f'models/{ticker}_cnnlstm.h5')

            # --- evaluate on test set (real metrics) ---
            y_pred_scaled = model.predict(X_test, verbose=0).flatten()
        else:
            # Fallback: simple linear trend for environments without TF
            trend = np.polyfit(range(len(arr.flatten())), arr.flatten(), 1)
            y_pred_scaled = np.polyval(trend, range(split, split + len(y_test)))
            y_pred_scaled = scaler.transform(y_pred_scaled.reshape(-1, 1)).flatten()

        y_pred_actual = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        metrics = _compute_metrics(y_test_actual, y_pred_actual)

        # Persist scaler
        import joblib
        joblib.dump(scaler, f'scalers/{ticker}_scaler.pkl')

        with _train_lock:
            _train_status[ticker] = {"status": "ready", "metrics": metrics}

        print(f"[StockSeer] {ticker} trained | RMSE={metrics['rmse']} | DirAcc={metrics['directional_accuracy']}%")

    except Exception as e:
        with _train_lock:
            _train_status[ticker] = {"status": "error", "metrics": None, "error": str(e)}
        print(f"[StockSeer] Training error for {ticker}: {e}")


def _get_model_and_scaler(ticker: str):
    """Load cached model + scaler from disk (returns None, None if not trained yet)."""
    model_path  = f'models/{ticker}_cnnlstm.h5'
    scaler_path = f'scalers/{ticker}_scaler.pkl'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    import joblib
    scaler = joblib.load(scaler_path)
    if TF_AVAILABLE:
        model = load_model(model_path)
    else:
        model = None
    return model, scaler


def predict_next_n(ticker: str, prices: list, n_days: int = 30):

    model, scaler = _get_model_and_scaler(ticker)

    # Not cached yet — check if training in progress
    status_info = _train_status.get(ticker, {})
    if model is None and status_info.get("status") != "training":
        # Kick off background training
        t = threading.Thread(target=_train_and_cache, args=(ticker, prices), daemon=True)
        t.start()
        return None, None, None  # tell caller to retry later

    if status_info.get("status") == "training":
        return None, None, None  # still training

    # Model ready — run multi-step prediction
    arr = np.array(prices, dtype=float).reshape(-1, 1)
    scaled = scaler.transform(arr)
    seq = list(scaled[-SEQ_LEN:].flatten())
    preds_scaled = []

    for _ in range(n_days):
        x = np.array(seq[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        if TF_AVAILABLE and model is not None:
            p = float(model.predict(x, verbose=0)[0, 0])
        else:
            p = float(seq[-1])  # fallback: last value
        preds_scaled.append(p)
        seq.append(p)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten().tolist()

    # Expanding confidence intervals (CI = ±1.5σ√k as per report)
    vol = float(np.std(np.diff(prices[-30:]) / np.array(prices[-30:-1])))
    upper = [round(p * (1 + vol * (i + 1) ** 0.5 * 1.5), 2) for i, p in enumerate(preds)]
    lower = [round(p * (1 - vol * (i + 1) ** 0.5 * 1.5), 2) for i, p in enumerate(preds)]

    return [round(p, 2) for p in preds], upper, lower


# ── REAL yfinance DATA ────────────────────────────────────────────────────────
_hist_cache = {}   # simple in-memory cache (ticker -> {data, ts})
CACHE_TTL   = 300  # seconds

def fetch_history(ticker: str, days: int = 365) -> dict:
    """Fetch OHLCV from Yahoo Finance; cache for CACHE_TTL seconds."""
    key = f"{ticker}_{days}"
    now = time.time()
    if key in _hist_cache and now - _hist_cache[key]['ts'] < CACHE_TTL:
        return _hist_cache[key]['data']

    try:
        df = yf.download(ticker, period=f"{days}d", auto_adjust=True, progress=False)
        if df.empty:
            return {}

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)
        info = yf.Ticker(ticker).info
        name = info.get('longName', info.get('shortName', ticker))

        result = {
            "dates":  df.index.strftime("%Y-%m-%d").tolist(),
            "open":   df['Open'].round(2).tolist(),
            "high":   df['High'].round(2).tolist(),
            "low":    df['Low'].round(2).tolist(),
            "close":  df['Close'].round(2).tolist(),
            "volume": df['Volume'].astype(int).tolist(),
            "name":   name,
            "ticker": ticker
        }
        _hist_cache[key] = {'data': result, 'ts': now}
        return result

    except Exception as e:
        print(f"[yfinance] Error fetching {ticker}: {e}")
        return {}


def fetch_live_price(ticker: str) -> dict:
    """Fetch current price using yfinance fast_info."""
    try:
        t    = yf.Ticker(ticker)
        fi   = t.fast_info
        price = float(getattr(fi, 'last_price', 0) or 0)
        prev  = float(getattr(fi, 'previous_close', price) or price)
        change = round(price - prev, 2)
        pct    = round((change / prev) * 100, 2) if prev else 0.0
        return {
            "ticker":      ticker,
            "price":       round(price, 2),
            "change":      change,
            "change_pct":  pct,
            "volume":      int(getattr(fi, 'three_month_average_volume', 0) or 0),
            "timestamp":   datetime.now().strftime("%H:%M:%S"),
            "high":        round(float(getattr(fi, 'day_high', price) or price), 2),
            "low":         round(float(getattr(fi, 'day_low', price) or price), 2)
        }
    except Exception as e:
        print(f"[yfinance live] {ticker}: {e}")
        return {"ticker": ticker, "price": 0, "change": 0, "change_pct": 0,
                "volume": 0, "timestamp": datetime.now().strftime("%H:%M:%S"),
                "high": 0, "low": 0}


# ── TECHNICAL INDICATORS (pure NumPy as in report) ───────────────────────────
def calc_sma(prices, w):
    return [round(float(np.mean(prices[max(0, i - w):i + 1])), 4) for i in range(len(prices))]

def calc_ema(prices, w):
    k, e = 2 / (w + 1), [float(prices[0])]
    for i in range(1, len(prices)):
        e.append(float(prices[i]) * k + e[-1] * (1 - k))
    return [round(x, 4) for x in e]

def calc_rsi(prices, w=14):
    diffs = np.diff(prices)
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    result = [None] * w
    for i in range(w, len(prices)):
        ag = np.mean(gains[i - w:i])
        al = np.mean(losses[i - w:i])
        result.append(round(100 - 100 / (1 + ag / al), 2) if al != 0 else 100.0)
    return result

def calc_bollinger(prices, w=20):
    mid   = calc_sma(prices, w)
    upper = [round(mid[i] + 2 * float(np.std(prices[max(0, i - w):i + 1])), 4) for i in range(len(prices))]
    lower = [round(mid[i] - 2 * float(np.std(prices[max(0, i - w):i + 1])), 4) for i in range(len(prices))]
    return upper, mid, lower


# ── AUTH HELPERS ──────────────────────────────────────────────────────────────
def require_auth(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json or {}
    users = load_users()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if username in users and users[username]['password'] == hash_password(password):
        session['user'] = username
        session['name'] = users[username]['name']
        return jsonify({"success": True, "name": users[username]['name'], "username": username})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json or {}
    users = load_users()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    name     = data.get('name', username)
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400
    if username in users:
        return jsonify({"success": False, "message": "Username already exists"}), 409
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be ≥ 6 characters"}), 400
    users[username] = {"password": hash_password(password), "name": name,
                       "created": str(datetime.now())}
    save_users(users)
    return jsonify({"success": True, "message": "Account created"})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/me')
def me():
    if 'user' in session:
        return jsonify({"logged_in": True, "username": session['user'], "name": session['name']})
    return jsonify({"logged_in": False})


# ── STOCK DATA ROUTES ─────────────────────────────────────────────────────────
SUPPORTED_TICKERS = [
    "AAPL","GOOGL","MSFT","TSLA","AMZN","NVDA","META","NFLX",
    "TCS.NS","INFY","RELIANCE.NS","HDFCBANK.NS"
]

@app.route('/api/stocks/list')
@require_auth
def stocks_list():
    return jsonify(SUPPORTED_TICKERS)

@app.route('/api/stocks/<ticker>/history')
@require_auth
def stock_history(ticker):
    ticker = ticker.upper()
    days   = int(request.args.get('days', 365))
    data   = fetch_history(ticker, days)
    if not data:
        return jsonify({"error": f"Could not fetch data for {ticker}"}), 404
    return jsonify(data)

@app.route('/api/stocks/<ticker>/live')
@require_auth
def stock_live(ticker):
    return jsonify(fetch_live_price(ticker.upper()))

@app.route('/api/stocks/<ticker>/train_status')
@require_auth
def train_status(ticker):
    """Let the frontend poll training progress."""
    ticker = ticker.upper()
    info   = _train_status.get(ticker, {"status": "not_started"})
    return jsonify(info)

@app.route('/api/stocks/<ticker>/predict')
@require_auth
def stock_predict(ticker):
    ticker = ticker.upper()
    n_days = int(request.args.get('days', 30))

    hist = fetch_history(ticker, 500)   # need ≥ SEQ_LEN + enough for training
    if not hist or len(hist.get('close', [])) < SEQ_LEN + 50:
        return jsonify({"error": "Not enough historical data"}), 400

    prices = hist['close']

    # Ensure model is trained (background thread if not)
    preds, upper, lower = predict_next_n(ticker, prices, n_days)

    if preds is None:
        # Training kicked off — tell frontend to poll
        return jsonify({
            "status": "training",
            "message": f"Model is being trained for {ticker}. Poll /api/stocks/{ticker}/train_status."
        }), 202

    # Future trading dates
    d     = datetime.now()
    count = 0
    future_dates = []
    while count < n_days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d.strftime("%Y-%m-%d"))
            count += 1

    metrics = _train_status.get(ticker, {}).get("metrics") or {}
    return jsonify({
        "ticker":            ticker,
        "predictions":       preds,
        "confidence_upper":  upper,
        "confidence_lower":  lower,
        "dates":             future_dates,
        "metrics":           metrics,
        "last_actual":       prices[-1]
    })

@app.route('/api/stocks/compare')
@require_auth
def compare_stocks():
    t1   = request.args.get('t1', 'AAPL').upper()
    t2   = request.args.get('t2', 'GOOGL').upper()
    days = int(request.args.get('days', 90))

    h1 = fetch_history(t1, days)
    h2 = fetch_history(t2, days)

    def normalize(prices):
        base = prices[0] if prices[0] != 0 else 1
        return [round(p / base * 100, 2) for p in prices]

    return jsonify({
        "stock1": {"ticker": t1, "name": h1.get("name", t1),
                   "prices": h1.get("close", []),
                   "normalized": normalize(h1["close"]) if h1.get("close") else [],
                   "dates": h1.get("dates", [])},
        "stock2": {"ticker": t2, "name": h2.get("name", t2),
                   "prices": h2.get("close", []),
                   "normalized": normalize(h2["close"]) if h2.get("close") else [],
                   "dates": h2.get("dates", [])}
    })

@app.route('/api/stocks/<ticker>/indicators')
@require_auth
def technical_indicators(ticker):
    ticker = ticker.upper()
    hist   = fetch_history(ticker, 200)
    if not hist:
        return jsonify({"error": "Data unavailable"}), 404

    prices = np.array(hist['close'], dtype=float)
    bb_u, bb_m, bb_l = calc_bollinger(prices)

    return jsonify({
        "dates":    hist['dates'],
        "close":    hist['close'],
        "volume":   hist['volume'],
        "sma20":    calc_sma(prices, 20),
        "sma50":    calc_sma(prices, 50),
        "ema20":    calc_ema(prices, 20),
        "rsi":      calc_rsi(prices),
        "bb_upper": bb_u,
        "bb_mid":   bb_m,
        "bb_lower": bb_l
    })


# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "tensorflow": TF_AVAILABLE,
                    "timestamp": str(datetime.now())})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  StockSeer — Real CNN+LSTM Stock Prediction System")
    print(f"  TensorFlow available : {TF_AVAILABLE}")
    print("  Default login        : admin / admin123")
    print("  URL                  : http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
