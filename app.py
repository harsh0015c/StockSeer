
from flask import Flask, request, jsonify, send_from_directory, session
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, hashlib, os, threading, warnings

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static')
app.secret_key = 'stock_pred_secret_2024'

# ─── LAZY TENSORFLOW IMPORT ───────────────────────────────────────────────────
# TF is imported once, only when the first prediction is requested,
# so the Flask server starts instantly even on slower machines.
_tf = None
_keras = None

def get_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        _tf = tf
        _keras = tf.keras
    return _tf, _keras

# ─── USER DATABASE ────────────────────────────────────────────────────────────
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    default = {
        "admin": {
            "password": hashlib.sha256("admin123".encode()).hexdigest(),
            "name": "Admin User",
            "created": str(datetime.now())
        }
    }
    save_users(default)
    return default

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()


# ─── REAL CNN+LSTM PREDICTOR ──────────────────────────────────────────────────
class CNNLSTMPredictor:
    """
    Real CNN+LSTM hybrid model.

    Architecture:
      Conv1D (64, k=3) → Conv1D (64, k=3) → MaxPool →
      LSTM (128, return_seq) → Dropout(0.2) →
      LSTM (64) → Dropout(0.2) →
      Dense(32) → Dense(1)

    Trained on the last `seq_len` closing prices (MinMax scaled).
    Models are saved to ./models/<TICKER>_cnnlstm.keras and reloaded on
    subsequent requests so you never retrain unnecessarily.
    """

    SEQ_LEN   = 60    # look-back window
    EPOCHS    = 60    # max epochs (EarlyStopping will cut short)
    BATCH     = 32

    def __init__(self):
        self.models  = {}   # ticker → keras model
        self.scalers = {}   # ticker → MinMaxScaler
        self._lock   = threading.Lock()
        os.makedirs('models', exist_ok=True)

    # ── model architecture ────────────────────────────────────────────────────
    def _build_model(self):
        tf, keras = get_tf()
        Sequential   = keras.models.Sequential
        Conv1D       = keras.layers.Conv1D
        MaxPooling1D = keras.layers.MaxPooling1D
        LSTM         = keras.layers.LSTM
        Dense        = keras.layers.Dense
        Dropout      = keras.layers.Dropout

        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu',
                   input_shape=(self.SEQ_LEN, 1)),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.Huber()   # robust to outliers
        )
        return model

    # ── data preparation ──────────────────────────────────────────────────────
    def _make_scaler(self, prices):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array(prices).reshape(-1, 1))
        return scaler

    def _make_sequences(self, scaled):
        X, y = [], []
        for i in range(self.SEQ_LEN, len(scaled)):
            X.append(scaled[i - self.SEQ_LEN:i])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    # ── train / load ──────────────────────────────────────────────────────────
    def _train(self, ticker, prices):
        tf, keras = get_tf()
        model_path = f'models/{ticker}_cnnlstm.keras'

        scaler = self._make_scaler(prices)
        self.scalers[ticker] = scaler
        scaled = scaler.transform(np.array(prices).reshape(-1, 1))

        # Load cached model if it exists
        if os.path.exists(model_path):
            self.models[ticker] = keras.models.load_model(model_path)
            return

        X, y = self._make_sequences(scaled)
        if len(X) < 10:
            raise ValueError(f"Not enough data to train for {ticker}")

        model = self._build_model()
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(
                patience=5, factor=0.5, monitor='val_loss')
        ]
        model.fit(
            X, y,
            epochs=self.EPOCHS,
            batch_size=self.BATCH,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
        )
        model.save(model_path)
        self.models[ticker] = model

    def ensure_trained(self, ticker, prices):
        with self._lock:
            if ticker not in self.models:
                self._train(ticker, prices)

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_next_n(self, ticker, prices, n_days=30):
        self.ensure_trained(ticker, prices)

        model  = self.models[ticker]
        scaler = self.scalers[ticker]

        scaled = scaler.transform(np.array(prices).reshape(-1, 1))
        seq    = list(scaled[-self.SEQ_LEN:].flatten())

        preds_scaled = []
        for _ in range(n_days):
            x    = np.array(seq[-self.SEQ_LEN:]).reshape(1, self.SEQ_LEN, 1)
            pred = float(model.predict(x, verbose=0)[0, 0])
            preds_scaled.append(pred)
            seq.append(pred)

        preds = scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)).flatten().tolist()

        # Expanding confidence interval based on historical volatility
        returns = np.diff(prices[-60:]) / np.array(prices[-60:-1])
        vol = float(np.std(returns))
        upper = [p * (1 + vol * (i + 1) ** 0.5 * 1.96) for i, p in enumerate(preds)]
        lower = [max(p * (1 - vol * (i + 1) ** 0.5 * 1.96), 0.01) for i, p in enumerate(preds)]

        return (
            [round(p, 2) for p in preds],
            [round(u, 2) for u in upper],
            [round(l, 2) for l in lower]
        )

    # ── walk-forward backtest metrics ─────────────────────────────────────────
    def get_metrics(self, ticker, prices):
        """
        Real walk-forward evaluation on the last 30 days of held-out data.
        """
        self.ensure_trained(ticker, prices)

        model  = self.models[ticker]
        scaler = self.scalers[ticker]
        scaled = scaler.transform(np.array(prices).reshape(-1, 1))

        # Use last 30 points as test set
        test_n = min(30, len(prices) - self.SEQ_LEN - 1)
        actuals, predictions = [], []

        for i in range(test_n):
            idx  = len(scaled) - test_n + i
            x    = scaled[idx - self.SEQ_LEN:idx].reshape(1, self.SEQ_LEN, 1)
            pred = float(model.predict(x, verbose=0)[0, 0])
            pred_price = float(scaler.inverse_transform([[pred]])[0, 0])
            actual_price = prices[idx]
            predictions.append(pred_price)
            actuals.append(actual_price)

        actuals     = np.array(actuals)
        predictions = np.array(predictions)
        mae  = float(np.mean(np.abs(actuals - predictions)))
        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        mape = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        da   = float(np.mean(
            np.sign(np.diff(actuals)) == np.sign(np.diff(predictions))
        ) * 100) if test_n > 1 else 0.0

        return {
            "mae":  round(mae,  2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r2":   round(r2,   4),
            "directional_accuracy": round(da, 1)
        }


predictor = CNNLSTMPredictor()


# ─── REAL MARKET DATA (yfinance) ──────────────────────────────────────────────
import yfinance as yf

# In-memory cache: ticker → {data, fetched_at}
_data_cache     = {}
_CACHE_SECONDS  = 300   # refresh every 5 minutes

def _cache_key(ticker, days):
    return f"{ticker}:{days}"

def generate_price_history(ticker, days=365):
    """
    Fetch REAL OHLCV data from Yahoo Finance via yfinance.
    Results are cached for 5 minutes to avoid hammering the API.
    Falls back to a minimal GBM stub only if yfinance is unavailable.
    """
    key = _cache_key(ticker, days)
    now = datetime.now()

    if key in _data_cache:
        cached = _data_cache[key]
        age = (now - cached['fetched_at']).total_seconds()
        if age < _CACHE_SECONDS:
            return cached['data']

    try:
        period = f"{days}d" if days <= 730 else "2y"
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty dataframe")

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)

        info = {}
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            pass

        result = {
            "dates":  df.index.strftime("%Y-%m-%d").tolist(),
            "open":   df['Open'].round(2).tolist(),
            "high":   df['High'].round(2).tolist(),
            "low":    df['Low'].round(2).tolist(),
            "close":  df['Close'].round(2).tolist(),
            "volume": df['Volume'].astype(int).tolist(),
            "name":   info.get('longName', ticker),
            "ticker": ticker
        }
        _data_cache[key] = {'data': result, 'fetched_at': now}
        return result

    except Exception as e:
        # Graceful fallback (simulation) if yfinance fails
        print(f"[WARN] yfinance failed for {ticker}: {e} — using simulation fallback")
        return _simulated_history(ticker, days)


def _simulated_history(ticker, days):
    """Emergency GBM fallback — only used when yfinance is unavailable."""
    CONFIGS = {
        "AAPL":  {"base": 178,  "vol": 0.015, "trend": 0.0003},
        "GOOGL": {"base": 141,  "vol": 0.018, "trend": 0.0002},
        "MSFT":  {"base": 415,  "vol": 0.013, "trend": 0.0004},
        "TSLA":  {"base": 245,  "vol": 0.035, "trend":-0.0001},
        "AMZN":  {"base": 185,  "vol": 0.020, "trend": 0.0003},
        "NVDA":  {"base": 875,  "vol": 0.030, "trend": 0.0008},
    }
    cfg = CONFIGS.get(ticker, {"base": 100, "vol": 0.02, "trend": 0.0002})
    np.random.seed(hash(ticker) % 2**31)
    price = cfg["base"]
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    end = datetime.now()
    for i in range(days, 0, -1):
        d = end - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        o = price
        ret = cfg["trend"] + cfg["vol"] * np.random.randn()
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(np.random.randn()) * 0.005)
        l = min(o, c) * (1 - abs(np.random.randn()) * 0.005)
        dates.append(d.strftime("%Y-%m-%d"))
        opens.append(round(o, 2)); highs.append(round(h, 2))
        lows.append(round(l, 2));  closes.append(round(c, 2))
        volumes.append(int(np.random.uniform(5e6, 50e6)))
        price = c
    return {"dates": dates, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": volumes,
            "name": ticker, "ticker": ticker}


def get_live_price(ticker):
    """Fetch the latest real-time quote from yfinance."""
    try:
        t  = yf.Ticker(ticker)
        info = t.fast_info
        price  = float(info.last_price)
        prev   = float(info.previous_close)
        change = round(price - prev, 2)
        pct    = round((change / prev) * 100, 2) if prev else 0
        return {
            "ticker":     ticker,
            "price":      round(price, 2),
            "change":     change,
            "change_pct": pct,
            "volume":     int(info.three_month_average_volume or 0),
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "high":       round(float(info.day_high or price), 2),
            "low":        round(float(info.day_low  or price), 2),
        }
    except Exception as e:
        print(f"[WARN] live price failed for {ticker}: {e}")
        # Fallback: use last close from history
        hist = generate_price_history(ticker, 5)
        last = hist["close"][-1] if hist["close"] else 100.0
        return {
            "ticker": ticker, "price": last,
            "change": 0, "change_pct": 0, "volume": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "high": last, "low": last,
        }


# ─── TECHNICAL INDICATORS (computed on real prices) ───────────────────────────

def compute_sma(prices, window):
    out = []
    for i in range(len(prices)):
        w = prices[max(0, i - window + 1): i + 1]
        out.append(round(float(np.mean(w)), 2))
    return out

def compute_ema(prices, window):
    k = 2 / (window + 1)
    e = [prices[0]]
    for i in range(1, len(prices)):
        e.append(prices[i] * k + e[-1] * (1 - k))
    return [round(x, 2) for x in e]

def compute_rsi(prices, window=14):
    arr = np.array(prices, dtype=float)
    diff = np.diff(arr)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    out = [None] * window
    for i in range(window, len(arr)):
        ag = np.mean(gains[i - window:i])
        al = np.mean(losses[i - window:i])
        if al == 0:
            out.append(100.0)
        else:
            rs = ag / al
            out.append(round(100 - 100 / (1 + rs), 2))
    return out

def compute_bollinger(prices, window=20):
    mid   = compute_sma(prices, window)
    upper, lower = [], []
    for i in range(len(prices)):
        w = prices[max(0, i - window + 1): i + 1]
        std = float(np.std(w))
        upper.append(round(mid[i] + 2 * std, 2))
        lower.append(round(mid[i] - 2 * std, 2))
    return upper, mid, lower

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    macd_line = [round(f - s, 4) for f, s in zip(ema_fast, ema_slow)]
    signal_line = compute_ema(macd_line, signal)
    histogram = [round(m - s, 4) for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


# ─── AUTH ─────────────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    data     = request.json
    users    = load_users()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if username in users and users[username]['password'] == hash_password(password):
        session['user'] = username
        session['name'] = users[username]['name']
        return jsonify({"success": True, "name": users[username]['name'], "username": username})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/register', methods=['POST'])
def register():
    data     = request.json
    users    = load_users()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    name     = data.get('name', username)
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400
    if username in users:
        return jsonify({"success": False, "message": "Username already exists"}), 409
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "created": str(datetime.now())
    }
    save_users(users)
    return jsonify({"success": True, "message": "Account created successfully"})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/me')
def me():
    if 'user' in session:
        return jsonify({"logged_in": True, "username": session['user'], "name": session['name']})
    return jsonify({"logged_in": False})


# ─── REQUIRE AUTH DECORATOR ───────────────────────────────────────────────────
def require_auth(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ─── STOCK DATA ROUTES ────────────────────────────────────────────────────────
SUPPORTED_TICKERS = [
    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
    "NVDA", "META", "NFLX",
    "TCS.NS", "INFY", "RELIANCE.NS", "HDFCBANK.NS"
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
    data   = generate_price_history(ticker, days)
    return jsonify(data)


@app.route('/api/stocks/<ticker>/live')
@require_auth
def stock_live(ticker):
    return jsonify(get_live_price(ticker.upper()))


@app.route('/api/stocks/<ticker>/predict')
@require_auth
def stock_predict(ticker):
    ticker = ticker.upper()
    n_days = int(request.args.get('days', 30))

    # Use 2 years of real data for training
    hist   = generate_price_history(ticker, 730)
    prices = hist['close']

    if len(prices) < predictor.SEQ_LEN + 10:
        return jsonify({"error": "Insufficient data for prediction"}), 400

    try:
        preds, upper, lower = predictor.predict_next_n(ticker, prices, n_days)
        metrics = predictor.get_metrics(ticker, prices)
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    # Generate future trading dates
    last_date    = datetime.now()
    future_dates = []
    d = last_date
    while len(future_dates) < n_days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d.strftime("%Y-%m-%d"))

    return jsonify({
        "ticker":           ticker,
        "predictions":      preds,
        "confidence_upper": upper,
        "confidence_lower": lower,
        "dates":            future_dates,
        "metrics":          metrics,
        "last_actual":      prices[-1]
    })


@app.route('/api/stocks/compare')
@require_auth
def compare_stocks():
    t1   = request.args.get('t1', 'AAPL').upper()
    t2   = request.args.get('t2', 'GOOGL').upper()
    days = int(request.args.get('days', 90))

    h1 = generate_price_history(t1, days)
    h2 = generate_price_history(t2, days)

    def normalize(prices):
        base = prices[0] if prices[0] != 0 else 1
        return [round(p / base * 100, 2) for p in prices]

    return jsonify({
        "stock1": {
            "ticker": t1, "name": h1["name"],
            "prices": h1["close"], "normalized": normalize(h1["close"]),
            "dates": h1["dates"]
        },
        "stock2": {
            "ticker": t2, "name": h2["name"],
            "prices": h2["close"], "normalized": normalize(h2["close"]),
            "dates": h2["dates"]
        }
    })


@app.route('/api/stocks/<ticker>/indicators')
@require_auth
def technical_indicators(ticker):
    """
    Returns all technical indicators computed on REAL price data.
    RSI, Bollinger Bands, SMA(20/50), EMA(20), MACD.
    """
    ticker = ticker.upper()
    hist   = generate_price_history(ticker, 365)

    if not hist['close']:
        return jsonify({"error": "No data available"}), 404

    prices  = hist['close']
    volumes = hist['volume']

    bb_upper, bb_mid, bb_lower = compute_bollinger(prices)
    macd_line, signal_line, histogram = compute_macd(prices)

    # Current values for quick summary
    last_rsi  = next((v for v in reversed(compute_rsi(prices)) if v is not None), None)
    last_price = prices[-1]
    last_bb_u  = bb_upper[-1]
    last_bb_l  = bb_lower[-1]

    # Overbought / Oversold signal
    rsi_signal = "Overbought" if last_rsi and last_rsi > 70 else (
                 "Oversold"   if last_rsi and last_rsi < 30 else "Neutral")
    bb_signal  = "Near Upper Band" if last_price > last_bb_u * 0.99 else (
                 "Near Lower Band" if last_price < last_bb_l * 1.01 else "Within Bands")

    return jsonify({
        "dates":       hist['dates'],
        "close":       prices,
        "volume":      volumes,
        # Moving averages
        "sma20":       compute_sma(prices, 20),
        "sma50":       compute_sma(prices, 50),
        "ema20":       compute_ema(prices, 20),
        # Momentum
        "rsi":         compute_rsi(prices),
        # Volatility
        "bb_upper":    bb_upper,
        "bb_mid":      bb_mid,
        "bb_lower":    bb_lower,
        # Trend
        "macd":        macd_line,
        "macd_signal": signal_line,
        "macd_hist":   histogram,
        # Summary signals
        "signals": {
            "rsi_value":  round(last_rsi, 2) if last_rsi else None,
            "rsi_signal": rsi_signal,
            "bb_signal":  bb_signal,
            "macd_cross": "Bullish" if macd_line[-1] > signal_line[-1] else "Bearish"
        }
    })


# ─── SERVE FRONTEND ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('models',  exist_ok=True)
    print("\n" + "=" * 58)
    print("  StockSeer | Real CNN+LSTM | http://localhost:5000")
    print("  Default login: admin / admin123")
    print("  NOTE: First prediction per ticker trains the model.")
    print("        Subsequent requests use the cached .keras file.")
    print("=" * 58 + "\n")
    app.run(debug=True, port=5000)
