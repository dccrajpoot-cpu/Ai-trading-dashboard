 ============================================================
# STEP 40 — AI WEB DASHBOARD
# Research / Paper Trading Prototype
# ============================================================

!pip -q install yfinance gradio scikit-learn matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# STOCK ANALYSIS FUNCTION
# ------------------------------------------------------------

def analyze_stock(ticker):

    try:
        # Download data
        raw = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if raw.empty:
            return "No data found.", None

        # Fix yfinance MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        data = raw.copy()

        # Required columns
        data = data[["Close", "Volume"]].copy()

        data["Close"] = pd.Series(
            data["Close"].to_numpy(),
            index=data.index
        )

        data["Volume"] = pd.Series(
            data["Volume"].to_numpy(),
            index=data.index
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        data["SMA20"] = data["Close"].rolling(20).mean()
        data["SMA50"] = data["Close"].rolling(50).mean()

        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        data["RSI"] = 100 - (100 / (1 + rs))

        data["Return_1D"] = data["Close"].pct_change()
        data["Return_5D"] = data["Close"].pct_change(5)

        data["Volatility"] = (
            data["Return_1D"].rolling(20).std()
        )

        data["Volume_Change"] = data["Volume"].pct_change()

        data["SMA20_Slope"] = data["SMA20"].pct_change(5)
        data["SMA50_Slope"] = data["SMA50"].pct_change(5)

        data["Price_vs_SMA20"] = (
            data["Close"] / data["SMA20"] - 1
        )

        data["Price_vs_SMA50"] = (
            data["Close"] / data["SMA50"] - 1
        )

        # ----------------------------------------------------
        # FUTURE TARGET
        # 3-day future return > 0.5% = UP
        # ----------------------------------------------------

        data["Future_Return_3D"] = (
            data["Close"].shift(-3) /
            data["Close"] - 1
        )

        data["Target"] = (
            data["Future_Return_3D"] > 0.005
        ).astype(int)

        features = [
            "SMA20",
            "SMA50",
            "RSI",
            "Return_1D",
            "Return_5D",
            "Volatility",
            "Volume_Change",
            "SMA20_Slope",
            "SMA50_Slope",
            "Price_vs_SMA20",
            "Price_vs_SMA50"
        ]

        clean = data.dropna(
            subset=features + ["Target"]
        ).copy()

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        X = clean[features]
        y = clean["Target"]

        split = int(len(clean) * 0.80)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # CURRENT PREDICTION
        # ----------------------------------------------------

        latest = data.dropna(
            subset=features
        ).iloc[-1]

        latest_X = pd.DataFrame(
            [latest[features].values],
            columns=features
        )

        probabilities = model.predict_proba(latest_X)[0]

        classes = list(model.classes_)

        if 1 in classes:
            up_prob = probabilities[
                classes.index(1)
            ]
        else:
            up_prob = 0

        down_prob = 1 - up_prob

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        if up_prob >= 0.40:
            signal = "BUY"
        elif up_prob <= 0.30:
            signal = "SELL"
        else:
            signal = "HOLD"

        price = float(latest["Close"])

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        chart_data = data.tail(180)

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            chart_data.index,
            chart_data["Close"],
            label="Price"
        )

        ax.plot(
            chart_data.index,
            chart_data["SMA20"],
            label="SMA20"
        )

        ax.plot(
            chart_data.index,
            chart_data["SMA50"],
            label="SMA50"
        )

        ax.set_title(
            f"{ticker} — AI Stock Analysis"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        AI STOCK ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stock       : {ticker}
Current Price: ${price:.2f}

UP Probability   : {up_prob * 100:.2f}%
DOWN Probability : {down_prob * 100:.2f}%

AI Signal : {signal}

Model Target:
3-Day Future Return > 0.5%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESEARCH / PAPER TRADING ONLY

This is an experimental AI model.
No profit is guaranteed.
Do not use this dashboard as financial advice.
"""

        return result, fig

    except Exception as e:

        return (
            f"ERROR:\n\n{str(e)}",
            None
        )


# ------------------------------------------------------------
# GRADIO DASHBOARD
# ------------------------------------------------------------

with gr.Blocks(
    title="AI Stock Trading Dashboard"
) as demo:

    gr.Markdown(
        """
# 📈 AI Stock Trading Dashboard

### Research / Paper Trading Prototype

Select a stock and run the AI analysis.
"""
    )

    stock = gr.Dropdown(
        choices=[
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "TSLA"
        ],
        value="AAPL",
        label="Select Stock"
    )

    run_button = gr.Button(
        "🤖 RUN AI ANALYSIS",
        variant="primary"
    )

    output = gr.Textbox(
        label="AI Result",
        lines=18
    )

    chart = gr.Plot(
        label="Price Chart"
    )

    run_button.click(
        fn=analyze_stock,
        inputs=stock,
        outputs=[output, chart]
    )

    gr.Markdown(
        """
⚠️ **Research / Paper Trading Only**

Built with Python + Gradio + Machine Learning.
"""
    )


# ------------------------------------------------------------
# START
# ------------------------------------------------------------

print("==============================================")
print("STEP 40 — AI WEB DASHBOARD")
print("==============================================")
print("Starting dashboard...")
print("")

demo.launch(
    share=True,
    debug=True
)
