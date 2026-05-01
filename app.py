
import os, json, hashlib, threading, warnings
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory, session
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static')
app.secret_key = 'stock_pred_secret_2024'
os.makedirs('models', exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  USER DATABASE
# ══════════════════════════════════════════════════════════════════════════════
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    default = {"admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Admin User", "created": str(datetime.now())
    }}
    save_users(default)
    return default

def save_users(u):
    with open(USERS_FILE, 'w') as f:
        json.dump(u, f, indent=2)

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  FAST ML PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
class FastStockPredictor:
    """
    Conceptual mapping to CNN+LSTM:

    CNN block  -> rolling window statistics at scales 5/10/20/30 days
                  (local pattern extraction, same as Conv1D filters)

    LSTM block -> lag returns at 1/2/3/5/7/10 days + cumulative momentum
                  (sequence memory, same as LSTM hidden state)

    Dense head -> GradientBoostingRegressor (non-linear regression)

    Training time:  <1 second on 500 samples
    Disk cache:     joblib, reloads in ~50 ms
    """

    LAG_STEPS   = [1, 2, 3, 5, 7, 10]
    ROLL_WINS   = [5, 10, 20, 30]
    MIN_SAMPLES = 60

    def __init__(self):
        self._pipelines = {}
        self._lock = threading.Lock()

    # ── feature engineering ───────────────────────────────────────────────────
    @staticmethod
    def _make_features(prices):
        s = pd.Series(np.array(prices, dtype=float))
        feats = {}

        # Lag returns (LSTM-like memory)
        for lag in FastStockPredictor.LAG_STEPS:
            feats[f'ret_{lag}d'] = s.pct_change(lag)

        # Multi-scale rolling stats (CNN-like local patterns)
        for w in FastStockPredictor.ROLL_WINS:
            feats[f'sma_{w}']    = s.rolling(w).mean()
            feats[f'std_{w}']    = s.rolling(w).std()
            feats[f'mom_{w}']    = s / s.shift(w) - 1
            feats[f'zscore_{w}'] = ((s - s.rolling(w).mean()) /
                                    (s.rolling(w).std() + 1e-9))

        # RSI(14) as a feature
        delta = s.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        feats['rsi14'] = 100 - 100 / (1 + gain / (loss + 1e-9))

        # Price position within 20-day range
        feats['hl_pos'] = ((s - s.rolling(20).min()) /
                           (s.rolling(20).max() - s.rolling(20).min() + 1e-9))

        # Absolute return (volatility proxy)
        feats['abs_ret'] = s.pct_change().abs()

        return pd.DataFrame(feats).dropna()

    # ── build training dataset ────────────────────────────────────────────────
    def _build_dataset(self, prices):
        arr = np.array(prices, dtype=float)
        df  = self._make_features(arr)
        # align prices array with feature rows
        price_aligned = arr[len(arr) - len(df):]
        y = np.diff(price_aligned) / price_aligned[:-1]  # next-day return
        X = df.values[:-1]
        return X, y

    # ── train / load ──────────────────────────────────────────────────────────
    def _train(self, ticker, prices):
        model_path = f'models/{ticker}_gb.pkl'
        if os.path.exists(model_path):
            self._pipelines[ticker] = joblib.load(model_path)
            return

        X, y = self._build_dataset(prices)
        if len(X) < self.MIN_SAMPLES:
            raise ValueError(f"Need at least {self.MIN_SAMPLES} rows, got {len(X)}")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                min_samples_leaf=5, random_state=42
            ))
        ])
        pipe.fit(X, y)
        joblib.dump(pipe, model_path)
        self._pipelines[ticker] = pipe

    def ensure_trained(self, ticker, prices):
        with self._lock:
            if ticker not in self._pipelines:
                self._train(ticker, prices)

    # ── autoregressive n-day prediction ──────────────────────────────────────
    def predict_next_n(self, ticker, prices, n_days=30):
        self.ensure_trained(ticker, prices)
        pipe     = self._pipelines[ticker]
        extended = list(prices)
        preds    = []

        for _ in range(n_days):
            df   = self._make_features(np.array(extended))
            if df.empty:
                break
            feat     = df.values[-1].reshape(1, -1)
            ret_pred = float(pipe.predict(feat)[0])
            ret_pred = max(min(ret_pred, 0.08), -0.08)    # clamp to ±8%
            next_p   = round(extended[-1] * (1 + ret_pred), 2)
            preds.append(next_p)
            extended.append(next_p)

        # Expanding confidence interval using historical volatility
        hist_ret  = np.diff(prices[-60:]) / np.array(prices[-60:-1], dtype=float)
        daily_vol = float(np.std(hist_ret))
        upper = [round(p * (1 + daily_vol * (i+1)**0.5 * 1.96), 2) for i, p in enumerate(preds)]
        lower = [round(max(p * (1 - daily_vol * (i+1)**0.5 * 1.96), 0.01), 2) for i, p in enumerate(preds)]
        return preds, upper, lower

    # ── walk-forward backtest ─────────────────────────────────────────────────
    def get_metrics(self, ticker, prices):
        self.ensure_trained(ticker, prices)
        pipe   = self._pipelines[ticker]
        arr    = np.array(prices, dtype=float)
        TEST_N = min(30, len(arr) // 5)
        actuals, predictions = [], []

        for i in range(TEST_N):
            end = len(arr) - TEST_N + i
            df  = self._make_features(arr[:end])
            if df.empty:
                continue
            feat = df.values[-1].reshape(1, -1)
            ret  = float(pipe.predict(feat)[0])
            pred = arr[end - 1] * (1 + ret)
            predictions.append(pred)
            actuals.append(float(arr[end]))

        if len(actuals) < 2:
            return {"mae": 0, "rmse": 0, "mape": 0, "r2": 0, "directional_accuracy": 0}

        a, p = np.array(actuals), np.array(predictions)
        mae  = float(np.mean(np.abs(a - p)))
        rmse = float(np.sqrt(np.mean((a - p)**2)))
        mape = float(np.mean(np.abs((a - p) / a)) * 100)
        ss_r = np.sum((a - p)**2)
        ss_t = np.sum((a - np.mean(a))**2)
        r2   = float(1 - ss_r / ss_t) if ss_t else 0.0
        da   = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100)
        return {"mae": round(mae,2), "rmse": round(rmse,2),
                "mape": round(mape,2), "r2": round(r2,4),
                "directional_accuracy": round(da,1)}


predictor = FastStockPredictor()


# ══════════════════════════════════════════════════════════════════════════════
#  REAL MARKET DATA  (yfinance + 5-min in-memory cache)
# ══════════════════════════════════════════════════════════════════════════════
_data_cache = {}
_CACHE_TTL  = 300   # seconds

def generate_price_history(ticker, days=365):
    key = f"{ticker}:{days}"
    now = datetime.now()
    if key in _data_cache:
        if (now - _data_cache[key]['ts']).total_seconds() < _CACHE_TTL:
            return _data_cache[key]['data']
    try:
        period = "2y" if days > 365 else f"{days}d"
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, threads=False)
        if df is None or df.empty:
            raise ValueError("empty")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        try:
            name = yf.Ticker(ticker).info.get('longName', ticker)
        except Exception:
            name = ticker
        result = {
            "dates":  df.index.strftime("%Y-%m-%d").tolist(),
            "open":   [round(float(v),2) for v in df['Open']],
            "high":   [round(float(v),2) for v in df['High']],
            "low":    [round(float(v),2) for v in df['Low']],
            "close":  [round(float(v),2) for v in df['Close']],
            "volume": [int(v) for v in df['Volume']],
            "name": name, "ticker": ticker,
        }
        _data_cache[key] = {'data': result, 'ts': now}
        return result
    except Exception as exc:
        print(f"[WARN] yfinance failed for {ticker}: {exc} — fallback")
        return _simulated_history(ticker, days)


def _simulated_history(ticker, days):
    CFG = {
        "AAPL": (178,0.015,0.0003),"GOOGL":(141,0.018,0.0002),
        "MSFT": (415,0.013,0.0004),"TSLA": (245,0.035,-0.0001),
        "AMZN": (185,0.020,0.0003),"NVDA": (875,0.030,0.0008),
        "META": (505,0.022,0.0005),"NFLX": (625,0.025,0.0002),
    }
    base, vol, trend = CFG.get(ticker, (100,0.02,0.0002))
    np.random.seed(abs(hash(ticker)) % 2**31)
    price = float(base)
    dates,opens,highs,lows,closes,vols = [],[],[],[],[],[]
    end = datetime.now()
    for i in range(days, 0, -1):
        d = end - timedelta(days=i)
        if d.weekday() >= 5: continue
        o = price; c = price*(1+trend+vol*np.random.randn())
        h = max(o,c)*(1+abs(np.random.randn())*0.005)
        l = min(o,c)*(1-abs(np.random.randn())*0.005)
        dates.append(d.strftime("%Y-%m-%d"))
        opens.append(round(o,2)); highs.append(round(h,2))
        lows.append(round(l,2)); closes.append(round(c,2))
        vols.append(int(np.random.uniform(5e6,50e6)))
        price = c
    return {"dates":dates,"open":opens,"high":highs,"low":lows,
            "close":closes,"volume":vols,"name":ticker,"ticker":ticker}


def get_live_price(ticker):
    try:
        fi    = yf.Ticker(ticker).fast_info
        price = round(float(fi.last_price), 2)
        prev  = round(float(fi.previous_close), 2)
        chg   = round(price - prev, 2)
        pct   = round(chg / prev * 100, 2) if prev else 0.0
        return {"ticker":ticker,"price":price,"change":chg,"change_pct":pct,
                "volume":int(getattr(fi,'three_month_average_volume',0) or 0),
                "timestamp":datetime.now().strftime("%H:%M:%S"),
                "high":round(float(fi.day_high or price),2),
                "low": round(float(fi.day_low  or price),2)}
    except Exception as exc:
        print(f"[WARN] live price {ticker}: {exc}")
        hist = generate_price_history(ticker, 5)
        last = hist["close"][-1] if hist["close"] else 100.0
        return {"ticker":ticker,"price":last,"change":0,"change_pct":0,
                "volume":0,"timestamp":datetime.now().strftime("%H:%M:%S"),
                "high":last,"low":last}


# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def _sma(prices, w):
    return [round(float(np.mean(prices[max(0,i-w+1):i+1])),4) for i in range(len(prices))]

def _ema(prices, w):
    k=2/(w+1); e=[float(prices[0])]
    for i in range(1,len(prices)):
        e.append(float(prices[i])*k+e[-1]*(1-k))
    return [round(x,4) for x in e]

def _rsi(prices, w=14):
    arr=np.array(prices,dtype=float); d=np.diff(arr)
    g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    out=[None]*w
    for i in range(w,len(arr)):
        ag=np.mean(g[i-w:i]); al=np.mean(l[i-w:i])
        out.append(round(100-100/(1+ag/al),2) if al!=0 else 100.0)
    return out

def _bollinger(prices, w=20):
    mid,upper,lower=[],[],[]
    for i in range(len(prices)):
        sl=prices[max(0,i-w+1):i+1]
        m=float(np.mean(sl)); s=float(np.std(sl))
        mid.append(round(m,4)); upper.append(round(m+2*s,4)); lower.append(round(m-2*s,4))
    return upper,mid,lower

def _macd(prices, fast=12, slow=26, signal=9):
    ef=_ema(prices,fast); es=_ema(prices,slow)
    line=[round(f-s,4) for f,s in zip(ef,es)]
    sig=_ema(line,signal)
    hist=[round(m-s,4) for m,s in zip(line,sig)]
    return line,sig,hist


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════
def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

@app.route('/api/login', methods=['POST'])
def login():
    d=request.json or {}; users=load_users()
    u=d.get('username','').strip(); p=d.get('password','')
    if u in users and users[u]['password']==hash_password(p):
        session['user']=u; session['name']=users[u]['name']
        return jsonify({"success":True,"name":users[u]['name'],"username":u})
    return jsonify({"success":False,"message":"Invalid credentials"}),401

@app.route('/api/register', methods=['POST'])
def register():
    d=request.json or {}; users=load_users()
    u=d.get('username','').strip(); p=d.get('password',''); name=d.get('name',u)
    if not u or not p:
        return jsonify({"success":False,"message":"Username and password required"}),400
    if u in users:
        return jsonify({"success":False,"message":"Username already exists"}),409
    if len(p)<6:
        return jsonify({"success":False,"message":"Password must be ≥6 characters"}),400
    users[u]={"password":hash_password(p),"name":name,"created":str(datetime.now())}
    save_users(users)
    return jsonify({"success":True,"message":"Account created"})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear(); return jsonify({"success":True})

@app.route('/api/me')
def me():
    if 'user' in session:
        return jsonify({"logged_in":True,"username":session['user'],"name":session['name']})
    return jsonify({"logged_in":False})


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK DATA ROUTES
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED = ["AAPL","GOOGL","MSFT","TSLA","AMZN","NVDA","META","NFLX",
             "TCS.NS","INFY","RELIANCE.NS","HDFCBANK.NS"]

@app.route('/api/stocks/list')
@require_auth
def stocks_list():
    return jsonify(SUPPORTED)

@app.route('/api/stocks/<ticker>/history')
@require_auth
def stock_history(ticker):
    days=int(request.args.get('days',365))
    return jsonify(generate_price_history(ticker.upper(),days))

@app.route('/api/stocks/<ticker>/live')
@require_auth
def stock_live(ticker):
    return jsonify(get_live_price(ticker.upper()))

@app.route('/api/stocks/<ticker>/predict')
@require_auth
def stock_predict(ticker):
    ticker=ticker.upper()
    n_days=int(request.args.get('days',30))
    hist=generate_price_history(ticker,730)
    prices=hist.get('close',[])
    if len(prices)<predictor.MIN_SAMPLES+10:
        return jsonify({"error":"Not enough price history for this ticker."}),400
    try:
        preds,upper,lower=predictor.predict_next_n(ticker,prices,n_days)
        metrics=predictor.get_metrics(ticker,prices)
    except Exception as exc:
        return jsonify({"error":str(exc)}),500
    future,d=[],datetime.now()
    while len(future)<n_days:
        d+=timedelta(days=1)
        if d.weekday()<5: future.append(d.strftime("%Y-%m-%d"))
    return jsonify({"ticker":ticker,"predictions":preds,
                    "confidence_upper":upper,"confidence_lower":lower,
                    "dates":future,"metrics":metrics,"last_actual":prices[-1]})

@app.route('/api/stocks/compare')
@require_auth
def compare_stocks():
    t1=request.args.get('t1','AAPL').upper()
    t2=request.args.get('t2','GOOGL').upper()
    days=int(request.args.get('days',90))
    h1=generate_price_history(t1,days); h2=generate_price_history(t2,days)
    def norm(px): b=px[0] or 1; return [round(p/b*100,2) for p in px]
    return jsonify({
        "stock1":{"ticker":t1,"name":h1["name"],"prices":h1["close"],
                  "normalized":norm(h1["close"]),"dates":h1["dates"]},
        "stock2":{"ticker":t2,"name":h2["name"],"prices":h2["close"],
                  "normalized":norm(h2["close"]),"dates":h2["dates"]}})

@app.route('/api/stocks/<ticker>/indicators')
@require_auth
def technical_indicators(ticker):
    ticker=ticker.upper()
    hist=generate_price_history(ticker,365)
    prices=hist.get('close',[])
    if not prices: return jsonify({"error":"No price data available"}),404
    bb_u,bb_m,bb_l=_bollinger(prices)
    macd_line,sig,macd_h=_macd(prices)
    rsi_vals=_rsi(prices)
    last_rsi=next((v for v in reversed(rsi_vals) if v is not None),50.0)
    last_p=prices[-1]
    rsi_sig=("Overbought" if last_rsi>70 else "Oversold" if last_rsi<30 else "Neutral")
    bb_sig=("Near Upper Band" if last_p>bb_u[-1]*0.99 else
            "Near Lower Band" if last_p<bb_l[-1]*1.01 else "Within Bands")
    return jsonify({
        "dates":hist["dates"],"close":prices,"volume":hist["volume"],
        "sma20":_sma(prices,20),"sma50":_sma(prices,50),"ema20":_ema(prices,20),
        "rsi":rsi_vals,"bb_upper":bb_u,"bb_mid":bb_m,"bb_lower":bb_l,
        "macd":macd_line,"macd_signal":sig,"macd_hist":macd_h,
        "signals":{"rsi_value":round(last_rsi,2),"rsi_signal":rsi_sig,
                   "bb_signal":bb_sig,
                   "macd_cross":"Bullish" if macd_line[-1]>sig[-1] else "Bearish"}})


# ══════════════════════════════════════════════════════════════════════════════
#  SERVE FRONTEND
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory('static','index.html')

if __name__=='__main__':
    os.makedirs('static',exist_ok=True)
    print("\n"+"═"*60)
    print("  StockSeer | Fast ML Predictor | http://localhost:5000")
    print("  Login: admin / admin123")
    print("  1st predict per ticker: ~1s  |  Cached: instant")
    print("═"*60+"\n")
    app.run(debug=True,port=5000)
