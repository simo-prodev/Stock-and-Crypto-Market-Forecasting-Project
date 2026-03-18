import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


print("\nIMPORTANT DISCLAIMER:")
print("This project is for educational and analytical purposes only.")
print("It does not provide financial advice or direct buy/sell recommendations.")
print("The analysis shared here is only an idea or market impression.")
print("You are solely responsible for your own trading and investment decisions.\n")


def download_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    needed_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[needed_cols].copy()
    df.dropna(inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()

    data["RSI14"] = compute_rsi(data["Close"], period=14)

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()

    rolling_std = data["Close"].rolling(20).std()
    data["BB_MID"] = data["SMA20"]
    data["BB_UPPER"] = data["SMA20"] + (2 * rolling_std)
    data["BB_LOWER"] = data["SMA20"] - (2 * rolling_std)

    data["Return_1d"] = data["Close"].pct_change(1)
    data["Return_5d"] = data["Close"].pct_change(5)
    data["Volatility_10"] = data["Return_1d"].rolling(10).std()
    data["Volume_Change"] = data["Volume"].pct_change()

    return data


def build_features(df: pd.DataFrame):
    data = df.copy()

    feature_cols = [
        "Close",
        "Volume",
        "SMA20",
        "SMA50",
        "EMA20",
        "RSI14",
        "MACD",
        "MACD_SIGNAL",
        "BB_UPPER",
        "BB_LOWER",
        "Return_1d",
        "Return_5d",
        "Volatility_10",
        "Volume_Change",
    ]

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    dataset = data[feature_cols + ["Target"]].dropna().copy()
    return dataset, feature_cols


def train_model(dataset: pd.DataFrame, feature_cols: list):
    split_index = int(len(dataset) * 0.8)

    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    X_train = train[feature_cols]
    y_train = train["Target"]

    X_test = test[feature_cols]
    y_test = test["Target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("=== Model Evaluation ===")
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    return model


def market_impression(latest_row: pd.Series, prob_up: float) -> dict:
    score = 0

    if latest_row["Close"] > latest_row["SMA20"]:
        score += 1
    else:
        score -= 1

    if latest_row["SMA20"] > latest_row["SMA50"]:
        score += 1
    else:
        score -= 1

    if latest_row["Close"] > latest_row["EMA20"]:
        score += 1
    else:
        score -= 1

    if latest_row["MACD"] > latest_row["MACD_SIGNAL"]:
        score += 1
    else:
        score -= 1

    if latest_row["RSI14"] > 55:
        score += 1
    elif latest_row["RSI14"] < 45:
        score -= 1

    if prob_up >= 0.60:
        score += 2
    elif prob_up <= 0.40:
        score -= 2

    if score >= 3:
        trend = "Bullish"
        explanation = "The market shows positive momentum and an upward bias."
    elif score <= -3:
        trend = "Bearish"
        explanation = "The market shows weakness and a downward bias."
    else:
        trend = "Neutral"
        explanation = "The market is mixed and does not show a strong directional edge."

    return {
        "score": score,
        "trend": trend,
        "prob_up": prob_up,
        "prob_down": 1 - prob_up,
        "explanation": explanation
    }


def plot_price(df: pd.DataFrame, symbol: str):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(df.index, df["SMA20"], label="SMA20")
    plt.plot(df.index, df["SMA50"], label="SMA50")
    plt.title(f"{symbol} Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_symbol(symbol: str, period: str = "2y", interval: str = "1d"):
    print(f"Downloading data for {symbol} ...")
    df = download_data(symbol, period=period, interval=interval)
    df = add_indicators(df)

    dataset, feature_cols = build_features(df)
    model = train_model(dataset, feature_cols)

    latest_features = dataset[feature_cols].iloc[[-1]]
    latest_row = df.loc[dataset.index[-1]]

    prob_up = model.predict_proba(latest_features)[0][1]
    result = market_impression(latest_row, prob_up)

    print("\n=== Latest Market Impression ===")
    print(f"Symbol       : {symbol}")
    print(f"Last Close   : {latest_row['Close']:.2f}")
    print(f"RSI(14)      : {latest_row['RSI14']:.2f}")
    print(f"MACD         : {latest_row['MACD']:.4f}")
    print(f"MACD Signal  : {latest_row['MACD_SIGNAL']:.4f}")
    print(f"SMA20        : {latest_row['SMA20']:.2f}")
    print(f"SMA50        : {latest_row['SMA50']:.2f}")
    print(f"Prob Up      : {result['prob_up']:.2%}")
    print(f"Prob Down    : {result['prob_down']:.2%}")
    print(f"Score        : {result['score']}")
    print(f"Impression   : {result['trend']}")
    print(f"Explanation  : {result['explanation']}")

    plot_price(df.tail(200), symbol)


if __name__ == "__main__":
    symbol = input("Enter symbol (example: AAPL or BTC-USD): ").strip().upper()
    analyze_symbol(symbol)
