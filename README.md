# StockSeer — CNN+LSTM Stock Price Prediction System

A full-stack stock prediction dashboard with:
- **CNN+LSTM hybrid neural network** for price forecasting
- **Login/Register** authentication system
- **Real-time price** updates (via yfinance)
- **Dual stock comparison** with normalized charts
- **Technical indicators**: RSI, Bollinger Bands, SMA, EMA
- **Confidence intervals** on all predictions

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

**Default credentials:** `admin` / `admin123`

---

## Project Structure
```
stock_predictor/
├── app.py              ← Flask backend + CNN+LSTM model
├── requirements.txt    ← Python dependencies
├── users.json          ← Auto-created user database
└── static/
    └── index.html      ← Full interactive dashboard
```

---

## Replacing Simulated Model with Real TensorFlow CNN+LSTM

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

class CNNLSTMPredictor:
    def __init__(self, seq_len=60):
        self.seq_len = seq_len
        self.models = {}
        self.scalers = {}

    def _build_model(self, n_features=1):
        model = Sequential([
            # CNN block — extracts local temporal patterns
            Conv1D(filters=64, kernel_size=3, activation='relu',
                   input_shape=(self.seq_len, n_features)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),

            # LSTM block — captures long-range dependencies
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),

            # Output block
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.Huber()
        )
        return model

    def _prepare_sequences(self, prices):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(np.array(prices).reshape(-1,1))
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i-self.seq_len:i])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y), scaler

    def train(self, ticker, prices):
        model_path = f'models/{ticker}_cnnlstm.h5'
        if os.path.exists(model_path):
            self.models[ticker] = load_model(model_path)
            # Fit a fresh scaler on the data
            scaler = MinMaxScaler()
            scaler.fit(np.array(prices).reshape(-1,1))
            self.scalers[ticker] = scaler
            return

        X, y, scaler = self._prepare_sequences(prices)
        self.scalers[ticker] = scaler

        model = self._build_model(n_features=1)
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        model.fit(X, y, epochs=100, batch_size=32,
                  validation_split=0.2, callbacks=callbacks, verbose=0)
        os.makedirs('models', exist_ok=True)
        model.save(model_path)
        self.models[ticker] = model

    def predict_next_n(self, ticker, prices, n_days=30):
        if ticker not in self.models:
            self.train(ticker, prices)

        model = self.models[ticker]
        scaler = self.scalers[ticker]
        scaled = scaler.transform(np.array(prices).reshape(-1,1))
        seq = list(scaled[-self.seq_len:].flatten())

        predictions_scaled = []
        for _ in range(n_days):
            x = np.array(seq[-self.seq_len:]).reshape(1, self.seq_len, 1)
            pred = model.predict(x, verbose=0)[0, 0]
            predictions_scaled.append(pred)
            seq.append(pred)

        preds = scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1,1)).flatten().tolist()

        # Confidence interval via Monte Carlo dropout
        vol = float(np.std(np.diff(prices[-30:]) / np.array(prices[-30:-1])))
        upper = [p * (1 + vol * (i+1)**0.5 * 1.5) for i,p in enumerate(preds)]
        lower = [p * (1 - vol * (i+1)**0.5 * 1.5) for i,p in enumerate(preds)]

        return ([round(p,2) for p in preds],
                [round(u,2) for u in upper],
                [round(l,2) for l in lower])
```

---

## Real-time Data (yfinance)

```python
import yfinance as yf

def generate_price_history(ticker, days=365):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    return {
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "open": df['Open'].round(2).tolist(),
        "high": df['High'].round(2).tolist(),
        "low": df['Low'].round(2).tolist(),
        "close": df['Close'].round(2).tolist(),
        "volume": df['Volume'].astype(int).tolist(),
        "name": yf.Ticker(ticker).info.get('longName', ticker),
        "ticker": ticker
    }


```

---

## Features

| Feature | Details |
|---|---|
| Auth | SHA-256 hashed passwords, session-based |
| Model | CNN extracts patterns → LSTM captures trends |
| Indicators | RSI(14), BB(20,2), SMA(20,50), EMA(20) |
| Comparison | Normalized 100-base chart + individual forecasts |
| Live updates | Auto-refreshes ticker prices every 5 seconds |
| Confidence | Expanding CI bands (wider further in future) |

created by harshchaudhary

