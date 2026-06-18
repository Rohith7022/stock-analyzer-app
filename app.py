# ============================================================
# Stock Analyzer - Financial Analytics Dashboard
# Personal Project | Python + Streamlit
# Author: Rohith Ravindra Reddy
# Features: GARCH volatility, VaR, MA20/MA50, live data
# Run: streamlit run app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")
st.title("📈 Stock Analyzer — Financial Analytics Dashboard")
st.markdown("*GARCH Volatility · Value-at-Risk · Moving Averages · Live Data*")

# ── SIDEBAR INPUTS ────────────────────────────────────────────
st.sidebar.header("Configuration")
ticker   = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period   = st.sidebar.selectbox("Period", ["1y","2y","3y","5y"], index=1)
var_conf = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
ma_short = st.sidebar.number_input("Short MA (days)", min_value=5,  max_value=50,  value=20)
ma_long  = st.sidebar.number_input("Long MA (days)",  min_value=20, max_value=200, value=50)

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(1))
    return df

with st.spinner(f"Loading {ticker} data..."):
    df = load_data(ticker, period)

if df.empty:
    st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    st.stop()

# ── KEY METRICS ───────────────────────────────────────────────
ret = df["Return"].dropna()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price",     f"${df['Close'].iloc[-1]:.2f}")
col2.metric("30-Day Return",     f"{ret.tail(21).add(1).prod()-1:.2%}")
col3.metric("Annual Vol",        f"{ret.std()*np.sqrt(252)*100:.1f}%")
col4.metric("Observations",      f"{len(df):,}")
col5.metric("Ticker",            ticker)

# ── PRICE CHART WITH MOVING AVERAGES ─────────────────────────
st.subheader(f"📊 {ticker} Price History with Moving Averages")
df[f"MA{ma_short}"] = df["Close"].rolling(ma_short).mean()
df[f"MA{ma_long}"]  = df["Close"].rolling(ma_long).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["Close"],           color="navy",   lw=1.2, label="Close Price", alpha=0.9)
ax.plot(df.index, df[f"MA{ma_short}"],   color="orange", lw=1.2, label=f"MA{ma_short}")
ax.plot(df.index, df[f"MA{ma_long}"],    color="red",    lw=1.2, label=f"MA{ma_long}")
ax.set_ylabel("Price ($)"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title(f"{ticker} — Closing Price with MA{ma_short} & MA{ma_long}")
st.pyplot(fig); plt.close()

# ── GARCH(1,1) VOLATILITY MODEL ──────────────────────────────
st.subheader("📉 GARCH(1,1) Conditional Volatility")
log_ret = df["LogReturn"].dropna() * 100  # in percent for ARCH library

garch = arch_model(log_ret, vol="Garch", p=1, q=1, dist="normal")
garch_fit = garch.fit(disp="off")

st.text(garch_fit.summary().tables[1].as_text())

cond_vol = garch_fit.conditional_volatility * np.sqrt(252)  # annualised

fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(df.index[-len(cond_vol):], cond_vol, color="crimson", lw=1.2)
ax2.set_ylabel("Annualised Conditional Volatility (%)")
ax2.set_title(f"{ticker} — GARCH(1,1) Conditional Volatility")
ax2.grid(alpha=0.3)
st.pyplot(fig2); plt.close()

# ── VALUE-AT-RISK (VaR) ───────────────────────────────────────
st.subheader(f"⚠️ Value-at-Risk (Historical, {var_conf*100:.0f}% Confidence)")

VaR = -np.percentile(ret.dropna(), (1-var_conf)*100)
CVaR = -ret[ret <= -VaR].mean()

col_v1, col_v2, col_v3 = st.columns(3)
col_v1.metric(f"1-Day VaR ({var_conf*100:.0f}%)", f"{VaR*100:.2f}%",
              help="Max expected daily loss with given confidence")
col_v2.metric("1-Day CVaR",      f"{CVaR*100:.2f}%",
              help="Expected loss beyond VaR (tail risk)")
col_v3.metric("10-Day VaR",      f"{VaR*np.sqrt(10)*100:.2f}%",
              help="10-day scaling via square-root-of-time")

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.hist(ret.dropna()*100, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
ax3.axvline(-VaR*100,  color="orange", lw=2, label=f"VaR = {VaR*100:.2f}%")
ax3.axvline(-CVaR*100, color="red",    lw=2, label=f"CVaR = {CVaR*100:.2f}%")
ax3.set_xlabel("Daily Return (%)"); ax3.set_ylabel("Frequency")
ax3.set_title(f"{ticker} — Return Distribution with VaR & CVaR")
ax3.legend(); ax3.grid(alpha=0.3)
st.pyplot(fig3); plt.close()

# ── RETURN DISTRIBUTION STATS ─────────────────────────────────
st.subheader("📋 Descriptive Statistics")
stats = ret.dropna().describe()
stats.loc["skewness"] = ret.skew()
stats.loc["kurtosis"] = ret.kurtosis()
st.dataframe(stats.to_frame("Daily Returns").style.format("{:.6f}"))

st.caption("Built by Rohith Ravindra Reddy | MS In Business Finance, VCU | github.com/Rohith7022")
