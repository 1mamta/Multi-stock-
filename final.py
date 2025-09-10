import streamlit as st
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================
# Helper Functions
# ===============================
def load_zip(zip_file):
    extract_folder = "Stocks"
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    else:
        for f in os.listdir(extract_folder):
            os.remove(os.path.join(extract_folder, f))

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    dfs = []
    for file in os.listdir(extract_folder):
        if file.endswith(".csv"):
            ticker = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(extract_folder, file))
            df.columns = [col.strip().lower() for col in df.columns]
            df.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adj close": "adj_close",
                "adjclose": "adj_close",
                "volume": "volume"
            }, inplace=True)
            df["Ticker"] = ticker
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def feature_engineering(df):
    if df.empty:
        return df
    df = df.sort_values(["Ticker", "date"])
    df["return"] = df.groupby("Ticker")["close"].pct_change()
    df["ma_7"] = df.groupby("Ticker")["close"].transform(lambda x: x.rolling(7).mean())
    df["ma_30"] = df.groupby("Ticker")["close"].transform(lambda x: x.rolling(30).mean())
    df["volatility"] = df.groupby("Ticker")["return"].transform(lambda x: x.rolling(7).std())
    df["target"] = (df.groupby("Ticker")["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

# Sector map for NIFTY50
sector_map = {
    "ADANIPORTS": "Infrastructure", "ASIANPAINT": "Consumer Goods",
    "AXISBANK": "Banking", "BAJAJ-AUTO": "Automobile",
    "BAJAJFINSV": "Financial Services", "BAJFINANCE": "Financial Services",
    "BHARTIARTL": "Telecom", "BPCL": "Energy", "BRITANNIA": "Consumer Goods",
    "CIPLA": "Pharma", "COALINDIA": "Energy", "DIVISLAB": "Pharma",
    "DRREDDY": "Pharma", "EICHERMOT": "Automobile", "GRASIM": "Cement",
    "HCLTECH": "IT", "HDFC": "Financial Services", "HDFCBANK": "Banking",
    "HEROMOTOCO": "Automobile", "HINDALCO": "Metals", "HINDUNILVR": "Consumer Goods",
    "ICICIBANK": "Banking", "INDUSINDBK": "Banking", "INFY": "IT",
    "IOC": "Energy", "ITC": "Consumer Goods", "JSWSTEEL": "Metals",
    "KOTAKBANK": "Banking", "LT": "Infrastructure", "M&M": "Automobile",
    "MARUTI": "Automobile", "NESTLEIND": "Consumer Goods", "NTPC": "Energy",
    "ONGC": "Energy", "POWERGRID": "Energy", "RELIANCE": "Energy",
    "SBIN": "Banking", "SHREECEM": "Cement", "SUNPHARMA": "Pharma",
    "TATAMOTORS": "Automobile", "TATASTEEL": "Metals", "TCS": "IT",
    "TECHM": "IT", "TITAN": "Consumer Goods", "ULTRACEMCO": "Cement",
    "UPL": "Chemicals", "WIPRO": "IT"
}

# ===============================
# Streamlit App
# ===============================
st.title("📈 Multi-Stock Sector Trend Analyzer (NIFTY50)")

uploaded_file = st.file_uploader("Upload ZIP of stock CSVs", type="zip")

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    data = load_zip(uploaded_file)

    if data.empty:
        st.error("⚠ No valid CSV files found in ZIP. Please check your dataset.")
    else:
        data["Sector"] = data["Ticker"].map(sector_map)

        st.subheader("📂 Sample Data")
        st.write(data.head())

        # Sector trends
        st.subheader("📊 Sector Trend Over Time")
        sector_trend = (
            data.groupby(["date", "Sector"])["close"].mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(10,5))
        sns.lineplot(data=sector_trend, x="date", y="close", hue="Sector", ax=ax)
        st.pyplot(fig)

        # Feature engineering + ML
        st.subheader("🤖 Machine Learning Prediction")
        data_ml = feature_engineering(data)

        if data_ml.empty or len(data_ml) < 50:
            st.warning("⚠ Not enough data for training ML model. Try uploading full dataset.")
        else:
            features = ["return", "ma_7", "ma_30", "volatility"]
            X = data_ml[features]
            y = data_ml["target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.text("📋 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=ax_cm, cmap="Blues")
            st.pyplot(fig_cm)

        # Investment Recommendation
        st.subheader("💡 Investment Recommendation")
        latest_date = data["date"].max()
        last_month = latest_date - pd.Timedelta(days=30)
        recent_data = data[data["date"] >= last_month].copy()

        if recent_data.empty:
            st.warning("⚠ Not enough recent data for recommendations.")
        else:
            recommendations = []
            for ticker, group in recent_data.groupby("Ticker"):
                avg_return = group["close"].pct_change().mean()
                volatility = group["close"].pct_change().std()
                ma_20 = group["close"].rolling(20).mean()
                trend_strength = (group["close"] > ma_20).mean()

                score = (avg_return * 100) - (volatility * 50) + (trend_strength * 10)
                recommendations.append([ticker, avg_return, volatility, trend_strength, score])

            rec_df = pd.DataFrame(recommendations, columns=["Ticker", "AvgReturn", "Volatility", "TrendStrength", "Score"])
            rec_df = rec_df.sort_values("Score", ascending=False).head(5)

            st.success("📌 Top 5 Stocks to Consider for Investment (last 30 days)")
            st.write(rec_df[["Ticker", "AvgReturn", "Volatility", "TrendStrength"]])

        # Stock selection
        st.subheader("🔍 Explore Individual Stock")
        stock_choice = st.selectbox("Choose a stock:", data["Ticker"].unique())
        stock_data = data[data["Ticker"] == stock_choice]

        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(stock_data["date"], stock_data["close"], label="Close Price")
        ax2.set_title(f"{stock_choice} Closing Price Over Time")
        ax2.legend()
        st.pyplot(fig2)