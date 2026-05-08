"""
GVC (Gamma-Vanna-Charm) Exposure & Signal Engine
──────────────────────────────────────────────────
Implements the full computation pipeline from the Master Quant Spec:
  - VolEngine: HV10 buffer, VIX feed, regime classification
  - GVC_Predictor: Vectorized Greeks, GEX/VEX/CEX exposure profiles
  - StrikeEngine: VRP-adjusted EM + skew/vanna/charm multipliers
  - SignalLayer: Composite go/no-go signal with suitability scoring
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time
from scipy.stats import norm
from datetime import datetime, date
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Set up a custom session to bypass basic yfinance rate-limiting
yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})


# ══════════════════════════════════════════════════════════════════
# Phase 1 — VolEngine + Core Data Pipeline
# ══════════════════════════════════════════════════════════════════

class VolEngine:
    """
    Maintains historical vol state needed by the EM formula.
    Run update_close() daily at 4:15 PM ET via scheduler.
    """

    def __init__(self, ticker: str = "XSP", lookback: int = 30):
        self.ticker = ticker
        self.lookback = lookback
        self.closes: deque = deque(maxlen=lookback)
        self.vix_prev_close: float = None
        self._initialize_history()

    def _initialize_history(self):
        """Seed the close buffer from yfinance on first run."""
        data = yf.download(self.ticker, period=f"{self.lookback + 5}d",
                           progress=False, session=yf_session)['Close']
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()
        for close in data.iloc[-self.lookback:]:
            self.closes.append(float(close))
        # Seed VIX prev close
        try:
            vix_hist = yf.download("^VIX", period="5d", progress=False, session=yf_session)['Close']
            if isinstance(vix_hist, pd.DataFrame):
                vix_hist = vix_hist.squeeze()
            if len(vix_hist) >= 2:
                self.vix_prev_close = float(vix_hist.iloc[-2])
        except Exception:
            pass

    def update_close(self, close_price: float):
        """Call daily at 4:15 PM ET. Appends today's close to the buffer."""
        self.closes.append(close_price)

    def hv10(self) -> float:
        """Annualized 10-day historical vol from daily log returns."""
        closes = np.array(list(self.closes))
        if len(closes) < 11:
            raise ValueError("Insufficient history for HV10 (need 11+ closes).")
        log_returns = np.diff(np.log(closes[-11:]))
        return float(np.std(log_returns, ddof=1) * np.sqrt(252))

    def bull_bear_regime(self) -> str:
        """20-day slope of closes → bull / neutral / bear."""
        closes = np.array(list(self.closes))
        if len(closes) < 20:
            return "neutral"
        slope = np.polyfit(range(20), closes[-20:], 1)[0]
        threshold = closes[-1] * 0.0005
        if slope > threshold:
            return "bull"
        elif slope < -threshold:
            return "bear"
        return "neutral"

    def fetch_vix_open(self) -> float:
        """Pull current VIX level from yfinance (used as IV proxy)."""
        vix = yf.Ticker("^VIX", session=yf_session)
        return float(vix.fast_info['last_price'])

    def classify_vanna_regime(self, vix_open: float) -> tuple:
        """
        Classify intraday vanna regime from VIX direction at open.
        Returns (regime_label, vanna_adj_scalar).
        """
        if self.vix_prev_close is None:
            return "flat", 1.00

        vix_chg = (vix_open - self.vix_prev_close) / self.vix_prev_close

        if vix_chg < -0.02:
            return "dropping", 0.92
        elif vix_chg > 0.02:
            return "spiking", 1.12
        return "flat", 1.00


# ══════════════════════════════════════════════════════════════════
# Phase 2 — GVC_Predictor (Greeks + Exposure Aggregation)
# ══════════════════════════════════════════════════════════════════

class GVC_Predictor:
    """
    Gamma-Vanna-Charm dealer exposure engine.
    Ingests options chain, computes vectorized Greeks, and produces
    GEX/VEX/CEX exposure profiles for regime classification.
    """

    def __init__(self, ticker: str = "XSP", r: float = 0.05):
        self.ticker = ticker
        self.r = r
        self.S = None
        self.chain = None
        self.df = None
        self.profile = {}
        self.by_strike = None

    def fetch_chain(self) -> pd.DataFrame:
        """
        Pull the full options chain from yfinance for front 1DTE/0DTE expiry.
        NOTE: yfinance OI is end-of-day — flag stale OI in live sessions.
        """
        tk = yf.Ticker(self.ticker, session=yf_session)
        self.S = tk.fast_info['last_price']
        expirations = tk.options
        today = date.today()
        rows = []

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            T = max((exp_date - today).days, 0) / 365.0

            if T > 5 / 365:
                continue

            try:
                chain = tk.option_chain(exp_str)
                time.sleep(0.5)  # Delay to prevent rate-limiting from Yahoo
            except Exception as e:
                print(f"[WARN] Chain fetch failed for {exp_str}: {e}")
                continue

            for opt_type, df_raw in [('call', chain.calls), ('put', chain.puts)]:
                df_raw = df_raw.copy()
                df_raw['type'] = opt_type
                df_raw['expiry'] = exp_str
                df_raw['T'] = T
                df_raw = df_raw.rename(columns={
                    'openInterest': 'oi',
                    'impliedVolatility': 'iv',
                })
                cols_to_keep = ['strike', 'type', 'expiry', 'oi', 'iv',
                                'bid', 'ask', 'T']
                available = [c for c in cols_to_keep if c in df_raw.columns]
                rows.append(df_raw[available])

        if not rows:
            raise ValueError(f"No chain data returned for {self.ticker}.")

        self.chain = pd.concat(rows, ignore_index=True)
        return self.chain

    def load_chain(self, df: pd.DataFrame):
        """Load a pre-built DataFrame (e.g., from historical snapshot)."""
        required = {'strike', 'type', 'oi', 'iv', 'T'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.chain = df.copy()

    def compute_greeks_vectorized(self) -> pd.DataFrame:
        """
        Compute Gamma, Vanna, Charm for every row via NumPy broadcasting.
        Drops rows with zero/NaN IV or OI before computation.
        """
        if self.chain is None:
            raise RuntimeError("No chain. Call fetch_chain() or load_chain() first.")
        if self.S is None:
            raise RuntimeError("Spot price not set.")

        df = self.chain.copy()
        df['oi'] = df['oi'].fillna(0).astype(float)
        df = df[(df['iv'] > 0) & (df['oi'] > 0) & (df['T'] > 0)].copy()

        S = self.S
        r = self.r
        K = df['strike'].values.astype(float)
        sigma = df['iv'].values.astype(float)
        T = df['T'].values.astype(float)

        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T
        # Avoid division by zero
        sigma_sqrt_T = np.where(sigma_sqrt_T == 0, 1e-10, sigma_sqrt_T)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        phi_d1 = norm.pdf(d1)

        # Greeks — do not modify these formulas
        gamma = phi_d1 / (S * sigma_sqrt_T)
        vanna = phi_d1 * (d2 / sigma)
        charm = -phi_d1 * (2 * r * T - d2 * sigma_sqrt_T) / (2 * T * sigma_sqrt_T)

        df['d1'], df['d2'] = d1, d2
        df['gamma'], df['vanna'], df['charm'] = gamma, vanna, charm

        self.df = df
        return self.df

    def aggregate_exposures(self) -> dict:
        """
        Convert per-option Greeks into $ dealer exposures (GEX/VEX/CEX).
        Computes all profile metrics needed by the signal layer.
        """
        if self.df is None:
            raise RuntimeError("Call compute_greeks_vectorized() first.")

        df = self.df.copy()
        S = self.S
        multiplier = 100

        df['GEX'] = df['gamma'] * df['oi'] * multiplier * S**2 * 0.01
        df.loc[df['type'] == 'put', 'GEX'] *= -1

        df['VEX'] = df['vanna'] * df['oi'] * multiplier * 0.01

        hours_remaining = (df['T'] * 252 * 6.5).clip(lower=0.01)
        df['CEX'] = df['charm'] * df['oi'] * multiplier / hours_remaining

        net_gex = df['GEX'].sum()
        total_abs_gex = df['GEX'].abs().sum()
        gex_strength = net_gex / total_abs_gex if total_abs_gex > 0 else 0.0

        by_strike = df.groupby('strike').agg(
            strike_gex=('GEX', 'sum'),
            strike_vex=('VEX', 'sum'),
            strike_cex=('CEX', 'sum'),
        ).reset_index().sort_values('strike')

        by_strike['cum_gex'] = by_strike['strike_gex'].cumsum()

        zero_gamma_strike = by_strike.loc[
            by_strike['cum_gex'].abs().idxmin(), 'strike'] if len(by_strike) > 0 else S
        pin_strike = by_strike.loc[
            by_strike['strike_gex'].abs().idxmax(), 'strike'] if len(by_strike) > 0 else S
        sign_changes = by_strike['cum_gex'].apply(np.sign).diff().abs() > 0
        gamma_walls = by_strike.loc[sign_changes, 'strike'].tolist()

        call_vex = df[df['type'] == 'call']['VEX'].sum()
        put_vex = df[df['type'] == 'put']['VEX'].sum()
        vanna_skew = call_vex - put_vex

        self.df = df
        self.by_strike = by_strike
        self.profile = {
            'net_gex': net_gex,
            'total_abs_gex': total_abs_gex,
            'gex_strength': gex_strength,
            'net_vex': df['VEX'].sum(),
            'net_cex': df['CEX'].sum(),
            'zero_gamma_strike': zero_gamma_strike,
            'pin_strike': pin_strike,
            'gamma_walls': gamma_walls,
            'vanna_skew': vanna_skew,
            'spot': S,
        }
        return self.profile


# ══════════════════════════════════════════════════════════════════
# Phase 3 — StrikeEngine
# ══════════════════════════════════════════════════════════════════

class StrikeEngine:
    """
    Computes the VRP-adjusted expected move and applies skew/vanna/charm
    multipliers to produce asymmetric short strikes for the iron condor.
    """

    def __init__(
        self,
        strike_increment: float = 5.0,
        skew_sensitivity: float = 0.50,
    ):
        self.strike_increment = strike_increment
        self.skew_sensitivity = skew_sensitivity

    def compute_expected_move(
        self,
        spot: float,
        vix: float,
        hv10: float,
    ) -> dict:
        """
        VRP-adjusted expected move. Averages IV-implied and realized-vol-implied
        1-day moves, then rounds up to the nearest strike increment.
        """
        IV = vix / 100 if vix > 1 else vix

        EM1 = spot * (IV / np.sqrt(252))
        EM2 = spot * (hv10 / np.sqrt(252))
        EM_avg = (EM1 + EM2) / 2
        EM_base = math.ceil(EM_avg / self.strike_increment) * self.strike_increment

        return {
            'EM1': round(EM1, 2),
            'EM2': round(EM2, 2),
            'EM_avg': round(EM_avg, 2),
            'EM_base': EM_base,
        }

    def compute_skew_adjustment(
        self,
        put_iv25: float,
        call_iv25: float,
    ) -> tuple:
        """
        Compute skew ratio and resulting asymmetry scalar.
        Returns (skew_adj, skew_label, skew_ratio).
        """
        skew_ratio = put_iv25 / call_iv25 if call_iv25 > 0 else 1.0
        skew_adj = (skew_ratio - 1.0) * self.skew_sensitivity

        if skew_ratio < 1.10:
            label = "FLAT"
        elif skew_ratio < 1.20:
            label = "NORMAL"
        elif skew_ratio < 1.35:
            label = "MODERATE"
        elif skew_ratio < 1.50:
            label = "ELEVATED"
        else:
            label = "EXTREME"

        return skew_adj, label, skew_ratio

    def compute_charm_adjustment(self, regime: str) -> float:
        """Bull/bear/neutral regime → call-side charm scalar."""
        return {'bull': 1.08, 'neutral': 1.00, 'bear': 0.95}.get(regime, 1.00)

    def compute_strikes(
        self,
        spot: float,
        vix: float,
        hv10: float,
        put_iv25: float,
        call_iv25: float,
        vanna_adj: float,
        charm_regime: str,
    ) -> dict:
        """
        Full strike computation pipeline.
        Returns all intermediates + final short strikes.
        """
        em = self.compute_expected_move(spot, vix, hv10)
        skew_adj, skew_label, skew_ratio = self.compute_skew_adjustment(
            put_iv25, call_iv25)
        charm_call_adj = self.compute_charm_adjustment(charm_regime)

        put_multiplier = (1 + skew_adj) * vanna_adj
        call_multiplier = (1 - skew_adj * 0.5) * vanna_adj * charm_call_adj

        put_distance = em['EM_base'] * put_multiplier
        call_distance = em['EM_base'] * call_multiplier

        inc = self.strike_increment
        short_put = math.floor((spot - put_distance) / inc) * inc
        short_call = math.ceil((spot + call_distance) / inc) * inc

        return {
            **em,
            'skew_ratio':       round(skew_ratio, 4),
            'skew_label':       skew_label,
            'skew_adj':         round(skew_adj, 4),
            'vanna_adj':        vanna_adj,
            'charm_call_adj':   charm_call_adj,
            'put_multiplier':   round(put_multiplier, 4),
            'call_multiplier':  round(call_multiplier, 4),
            'put_distance':     round(put_distance, 2),
            'call_distance':    round(call_distance, 2),
            'short_put':        short_put,
            'short_call':       short_call,
        }


# ══════════════════════════════════════════════════════════════════
# Phase 4 — SignalLayer
# ══════════════════════════════════════════════════════════════════

class SignalLayer:
    """
    Applies all pre-trade filters and produces the composite go/no-go signal.
    """

    def __init__(
        self,
        vix_lower: float = 11.0,
        vix_upper: float = 30.0,
        gex_positive_threshold: float = 5e9,
        gex_shrink_factor: float = 0.70,
        low_vex_threshold: float = 1e6,
        moderate_cex_threshold: float = 1e5,
        gex_proximity_pct: float = 0.003,
    ):
        self.vix_lower = vix_lower
        self.vix_upper = vix_upper
        self.gex_positive_threshold = gex_positive_threshold
        self.gex_shrink_factor = gex_shrink_factor
        self.low_vex_threshold = low_vex_threshold
        self.moderate_cex_threshold = moderate_cex_threshold
        self.gex_proximity_pct = gex_proximity_pct

    def evaluate(
        self,
        vix: float,
        strike_result: dict,
        gvc_profile: dict,
        time_to_close_hours: float,
        condor_breakeven_width: float,
        top_gex_strikes: list = None,
    ) -> dict:
        """
        Run all filters and produce composite signal.
        """
        S = gvc_profile['spot']
        net_gex = gvc_profile['net_gex']
        net_vex = gvc_profile['net_vex']
        net_cex = gvc_profile['net_cex']
        skew_ratio = strike_result['skew_ratio']
        skew_label = strike_result['skew_label']
        vanna_adj = strike_result['vanna_adj']
        short_put = strike_result['short_put']
        short_call = strike_result['short_call']
        EM_base = strike_result['EM_base']

        flags = []
        skip = False
        high_risk = False

        # --- Hard gates ---
        if not (self.vix_lower < vix < self.vix_upper):
            flags.append(f"VIX out of range ({vix:.1f})")
            skip = True

        if skew_ratio > 1.50:
            flags.append(f"Extreme skew ({skew_ratio:.3f})")
            skip = True

        # --- High risk ---
        if vanna_adj > 1.05 and skew_ratio > 1.35:
            flags.append("VIX spiking + elevated skew")
            high_risk = True

        # --- GEX proximity ---
        gex_warning = False
        if top_gex_strikes:
            gex_warning = any(
                abs(S - g) / S < self.gex_proximity_pct
                for g in top_gex_strikes
            )
            if gex_warning:
                flags.append("Spot near GEX concentration node")

        # --- Regime classification ---
        if net_gex > self.gex_positive_threshold:
            regime = "HIGH_STABILITY"
            adjusted_em = EM_base * self.gex_shrink_factor
        elif net_gex > 0:
            regime = "MILD_POSITIVE"
            adjusted_em = EM_base * 0.90
        else:
            regime = "NEGATIVE_GEX"
            adjusted_em = EM_base

        predicted_range_width = 2 * adjusted_em

        # --- Suitability score ---
        suitability = 0
        if net_gex > self.gex_positive_threshold:
            suitability += 40
        elif net_gex > 0:
            suitability += 15
        if abs(net_vex) < self.low_vex_threshold:
            suitability += 25
        if abs(net_cex) < self.moderate_cex_threshold and time_to_close_hours < 3:
            suitability += 20
        if predicted_range_width < condor_breakeven_width * 0.85:
            suitability += 15

        # --- Determine action ---
        if skip:
            signal = "SKIP"
            action = "Do not trade. Log reason."
            size_multiplier = 0.0
        elif high_risk:
            signal = "HIGH_RISK"
            action = "Widen strikes by 1.5×, or skip."
            size_multiplier = 0.5
            short_put = math.floor(short_put * 0.9 / 5) * 5
            short_call = math.ceil(short_call * 1.1 / 5) * 5
        elif gex_warning:
            signal = "CAUTION"
            action = "Trade at 50% size. Widen 1 strike increment."
            size_multiplier = 0.5
        elif suitability >= 85:
            signal = "STRONG"
            action = "High-confidence condor. Full size."
            size_multiplier = 1.0
        elif suitability >= 60:
            signal = "MODERATE"
            action = "Acceptable. Size conservatively."
            size_multiplier = 0.75
        elif suitability >= 35:
            signal = "WEAK"
            action = "Marginal. Consider passing."
            size_multiplier = 0.5
        else:
            signal = "AVOID"
            action = "Conditions unfavorable for short vol."
            size_multiplier = 0.0

        return {
            'signal':               signal,
            'action':               action,
            'regime':               regime,
            'suitability':          suitability,
            'size_multiplier':      size_multiplier,
            'flags':                flags,
            'short_put':            short_put,
            'short_call':           short_call,
            'adjusted_em':          round(adjusted_em, 2),
            'predicted_range':      (round(S - adjusted_em, 2),
                                     round(S + adjusted_em, 2)),
            'net_gex_B':            round(net_gex / 1e9, 2),
            'skew_label':           skew_label,
            'vanna_regime':         'spiking' if vanna_adj > 1.05
                                    else 'dropping' if vanna_adj < 0.95
                                    else 'flat',
        }
