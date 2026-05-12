import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="XSP 1DTE Condor Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Global App Typography & Aesthetics */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Glassmorphism for st.metric */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

div[data-testid="stMetricLabel"] > div > div > p {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8892b0 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] > div {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #e6f1ff !important;
}

div[data-testid="stMetricDelta"] > div {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    margin-top: 2px;
}

/* Custom header gradients */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    background: -webkit-linear-gradient(45deg, #4da6ff, #ff3366);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# SIDEBAR — MODE TOGGLE
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔀 Calculator Mode")
    mode = st.radio(
        "Select calculator",
        options=["VRP Scanner", "GVC Signal Engine"],
        index=0,
        help="Toggle between the classic VRP scanner and the new GVC exposure engine",
        label_visibility="collapsed",
    )
    st.markdown("---")

# ────────────────────────────────────────────────
# ROUTE TO SELECTED MODE
# ────────────────────────────────────────────────
if mode == "GVC Signal Engine":
    from gvc_page import render_gvc_page
    render_gvc_page()
    st.stop()

# ══════════════════════════════════════════════════════════════════
# ORIGINAL VRP SCANNER — unchanged logic below
# ══════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
DATA_TICKER = "SPY"  # options chain & IV source (yfinance lacks XSP options)
SPOT_TICKER = "^XSP"  # actual XSP index price for accurate strike placement
DISPLAY_TICKER = "XSP"  # what you trade (European, cash-settled, no assignment risk)
RV_WINDOW = 10
WING_WIDTH = 1.0
TARGET_COLLATERAL = 100
MIN_CREDIT = 0.20

# ────────────────────────────────────────────────
# CACHE FUNCTIONS (faster reloads)
# ────────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_nearest_expiration(ticker):
    t = yf.Ticker(ticker)
    expirations = t.options
    today = datetime.now().date()
    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if tomorrow in expirations:
        return tomorrow, 1

    future_exps = [
        exp
        for exp in expirations
        if datetime.strptime(exp, "%Y-%m-%d").date() > today
    ]
    if future_exps:
        nearest = min(
            future_exps,
            key=lambda x: abs(
                (datetime.strptime(x, "%Y-%m-%d").date() - today).days
            ),
        )
        dte = (datetime.strptime(nearest, "%Y-%m-%d").date() - today).days
        return nearest, dte
    return None, None


@st.cache_data(ttl=900)
def calculate_rv(ticker, window):
    data = yf.download(ticker, period=f"{window * 2}d", progress=False)
    close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = data[close_col]
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    logrets = np.log(prices / prices.shift(1)).dropna().iloc[-window:]
    daily_std = logrets.std()
    return daily_std * np.sqrt(252) * 100


@st.cache_data(ttl=900)
def get_atm_iv_and_spot(ticker, expiration, spot_ticker=None):
    spot_src = yf.Ticker(spot_ticker or ticker)
    hist = spot_src.history(period="1d")["Close"]
    if isinstance(hist, pd.DataFrame):
        hist = hist.squeeze()
    spot = float(hist.iloc[-1])

    t = yf.Ticker(ticker)
    chain = t.option_chain(expiration)
    calls = chain.calls.reset_index(drop=True)
    puts = chain.puts.reset_index(drop=True)

    strikes = calls["strike"].values
    atm_strike = float(strikes[np.argmin(np.abs(strikes - spot))])

    call_iv = calls.loc[calls["strike"] == atm_strike, "impliedVolatility"].values
    put_iv = puts.loc[puts["strike"] == atm_strike, "impliedVolatility"].values

    if len(call_iv) > 0 and len(put_iv) > 0:
        blended = (call_iv[0] + put_iv[0]) / 2 * 100
    elif len(call_iv) > 0:
        blended = call_iv[0] * 100
    elif len(put_iv) > 0:
        blended = put_iv[0] * 100
    else:
        blended = np.nan

    return spot, blended, atm_strike


# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
st.title("XSP 1DTE VRP-Adjusted Iron Condor Scanner")
st.caption(
    "Run at ~12:30 PM PST • XSP (cash-settled, no assignment risk) • SPY data as proxy • VRP = ATM IV − 10D RV • 1-pt wings"
)

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("Refresh Data Now", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col2:
    auto_refresh = st.checkbox("Auto-refresh every 5 min", value=False)

if auto_refresh:
    st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

# ────────────────────────────────────────────────
# CORE LOGIC
# ────────────────────────────────────────────────
with st.spinner("Fetching live data…"):
    exp, dte = get_nearest_expiration(DATA_TICKER)

    if exp is None:
        st.error("No short-dated expiration found. Market closed?")
    else:
        spot, atm_iv, atm_strike = get_atm_iv_and_spot(DATA_TICKER, exp, SPOT_TICKER)

        if np.isnan(atm_iv):
            st.error("Could not read ATM IV from chain.")
        else:
            rv = calculate_rv(DATA_TICKER, RV_WINDOW)
            vrp = atm_iv - rv

            time_frac = dte / 365.0
            sqrt_t = np.sqrt(time_frac)

            em1_pct = atm_iv / 100 * sqrt_t * 100
            em1_dol = spot * (em1_pct / 100)

            vrp_adj_iv = max(atm_iv - vrp, 5)
            em2_pct = vrp_adj_iv / 100 * sqrt_t * 100
            em2_dol = spot * (em2_pct / 100)

            adj_em_pct = (em1_pct + em2_pct) / 2
            adj_em_dol = (em1_dol + em2_dol) / 2

            # ────────────────────────────────────────────────
            # ASYMMETRIC PROBABILITY (from OTM put/call IV skew)
            # ────────────────────────────────────────────────
            # ATM put/call IVs at the SAME strike are nearly identical
            # (put-call parity). Real skew lives in OTM options.
            # Use ~5% OTM strikes to capture the true skew surface.
            t_chain = yf.Ticker(DATA_TICKER)
            chain_data = t_chain.option_chain(exp)

            otm_put_target = spot * 0.95   # 5% OTM put
            otm_call_target = spot * 1.05  # 5% OTM call

            put_strikes = chain_data.puts['strike'].values
            call_strikes = chain_data.calls['strike'].values

            # Find nearest OTM put strike and its IV
            if len(put_strikes) > 0:
                otm_put_strike = float(put_strikes[np.argmin(np.abs(put_strikes - otm_put_target))])
                otm_put_iv_raw = chain_data.puts.loc[
                    chain_data.puts['strike'] == otm_put_strike, 'impliedVolatility'
                ].values
                otm_put_iv = float(otm_put_iv_raw[0]) if len(otm_put_iv_raw) > 0 else 0.20
            else:
                otm_put_strike = spot * 0.95
                otm_put_iv = 0.20

            # Find nearest OTM call strike and its IV
            if len(call_strikes) > 0:
                otm_call_strike = float(call_strikes[np.argmin(np.abs(call_strikes - otm_call_target))])
                otm_call_iv_raw = chain_data.calls.loc[
                    chain_data.calls['strike'] == otm_call_strike, 'impliedVolatility'
                ].values
                otm_call_iv = float(otm_call_iv_raw[0]) if len(otm_call_iv_raw) > 0 else 0.18
            else:
                otm_call_strike = spot * 1.05
                otm_call_iv = 0.18

            # Directional probabilities from OTM put/call IV ratio
            # Higher OTM put IV → market pricing more downside risk
            total_iv = otm_put_iv + otm_call_iv
            if total_iv > 0:
                p_down = otm_put_iv / total_iv
                p_up = otm_call_iv / total_iv
            else:
                p_down = 0.50
                p_up = 0.50

            skew_ratio = otm_put_iv / otm_call_iv if otm_call_iv > 0 else 1.0

            # Asymmetric EM: scale by directional probability
            em_down = adj_em_dol * (2 * p_down)
            em_up = adj_em_dol * (2 * p_up)

            # GEX compression/expansion from skew (±10%)
            GEX_COMPRESSION = 0.10
            MIN_SKEW_RATIO = 1.02
            if skew_ratio >= MIN_SKEW_RATIO:
                # Bearish skew: -GEX downside (expand), +GEX upside (compress)
                em_down_final = em_down * (1 + GEX_COMPRESSION)
                em_up_final = em_up * (1 - GEX_COMPRESSION)
                bias_label = "🐻 BEARISH"
            elif 1.0 / skew_ratio >= MIN_SKEW_RATIO:
                # Bullish skew: +GEX downside (compress), -GEX upside (expand)
                em_down_final = em_down * (1 - GEX_COMPRESSION)
                em_up_final = em_up * (1 + GEX_COMPRESSION)
                bias_label = "🐂 BULLISH"
            else:
                em_down_final = em_down
                em_up_final = em_up
                bias_label = "⚖️ NEUTRAL"

            # Short strikes using asymmetric EM
            lower_short = round(spot - em_down_final * 0.98)
            upper_short = round(spot + em_up_final * 0.98)
            lower_long = lower_short - WING_WIDTH
            upper_long = upper_short + WING_WIDTH

            is_load_day = vrp > 5

            # ────────────────────────────────────────────────
            # SIGNAL
            # ────────────────────────────────────────────────
            if is_load_day:
                st.error("🚫 LOAD DAY — VRP > 5 — DO NOT TRADE TODAY")
                st.caption(
                    "High VRP days historically have a 47% win rate. "
                    "The market is pricing in a big move. Sit this one out."
                )
                st.markdown("---")

            # ────────────────────────────────────────────────
            # DISPLAY
            # ────────────────────────────────────────────────
            st.markdown("### Live Snapshot")
            cols = st.columns(4)
            cols[0].metric("Spot", f"${spot:.2f}")
            cols[1].metric("ATM IV (blended)", f"{atm_iv:.2f}%")
            cols[2].metric("10D RV", f"{rv:.2f}%")
            cols[3].metric(
                "VRP",
                f"{vrp:+.2f}%",
                delta_color="normal" if vrp > 0 else "inverse",
            )

            st.markdown("### Expected Move Calculation")
            em_cols = st.columns(3)
            em_cols[0].metric("EM (IV)", f"${em1_dol:.2f}", help=f"Spot × IV × √T = {em1_pct:.2f}%")
            em_cols[1].metric("EM (IV − VRP)", f"${em2_dol:.2f}", help=f"Spot × (IV−VRP) × √T = {em2_pct:.2f}%")
            em_cols[2].metric("Avg EM (Symmetric)", f"${adj_em_dol:.2f}", help=f"(EM1 + EM2) / 2 = {adj_em_pct:.2f}%")

            # Directional Probability display
            st.markdown("### 🎲 Directional Probability")

            prob_cols = st.columns(3)
            prob_cols[0].metric("⬇️ P(Down)", f"{p_down:.1%}")
            prob_cols[1].metric("⬆️ P(Up)", f"{p_up:.1%}")
            prob_cols[2].metric("Skew Bias", bias_label)

            # Asymmetric EM display
            st.markdown("### 📐 Asymmetric Expected Move")
            asym_cols = st.columns(3)
            asym_cols[0].metric("EM ⬇️ (Downside)", f"${em_down_final:.2f}",
                                help="Avg EM × 2×P(down) × GEX adj")
            asym_cols[1].metric("EM ⬆️ (Upside)", f"${em_up_final:.2f}",
                                help="Avg EM × 2×P(up) × GEX adj")
            asym_cols[2].metric("Skew Ratio", f"{skew_ratio:.4f}",
                                help=f"OTM Put IV / OTM Call IV = {otm_put_iv:.4f} / {otm_call_iv:.4f}")

            st.markdown(
                f"**Expected range**: ${spot - em_down_final:.0f} – ${spot + em_up_final:.0f}"
            )

            st.markdown(f"### Suggested {DISPLAY_TICKER} Iron Condor (1-pt wings)")
            st.code(
                f"Sell Put  {lower_short:.0f}  /  Buy Put  {lower_long:.0f}\n"
                f"Sell Call {upper_short:.0f}  /  Buy Call {upper_long:.0f}",
                language="text",
            )
            st.caption(f"Trade {DISPLAY_TICKER} options at these strikes • Check your broker for actual credit — aim for $0.30+ mid price")

            if not is_load_day:
                st.success("✅ TAKE THIS TRADE — backtested 72% WR on non-LOAD days")

            st.caption(
                "⚠️ _Directional probabilities and asymmetric EM are model-derived estimates "
                "from the ATM IV skew, not guaranteed outcomes. Not financial advice._"
            )
