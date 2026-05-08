"""
GVC Signal Engine — Streamlit Dashboard Page
─────────────────────────────────────────────
Renders the full Gamma-Vanna-Charm exposure dashboard with:
  - GEX bar profile by strike
  - Cumulative GEX curve
  - Suitability gauge
  - Signal banner + strike recommendations
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from gvc_engine import VolEngine, GVC_Predictor, StrikeEngine, SignalLayer


def _signal_color(signal: str) -> str:
    return {
        'STRONG': '#00cc55', 'MODERATE': '#88cc00', 'WEAK': '#ccaa00',
        'CAUTION': '#cc6600', 'HIGH_RISK': '#cc3300', 'SKIP': '#cc0000',
        'AVOID': '#880000',
    }.get(signal, '#888888')


def _regime_emoji(regime: str) -> str:
    return {
        'HIGH_STABILITY': '🟢', 'MILD_POSITIVE': '🟡', 'NEGATIVE_GEX': '🔴',
    }.get(regime, '⚪')


@st.cache_data(ttl=900)
def _run_gvc_pipeline(ticker: str, strike_inc: float):
    """Fetch data and run the full GVC pipeline. Cached for 15 min."""
    import time

    vol = VolEngine(ticker=ticker)
    pred = GVC_Predictor(ticker=ticker)
    se = StrikeEngine(strike_increment=strike_inc)
    sig = SignalLayer()

    # Retry with exponential backoff for Yahoo rate-limits
    max_retries = 3
    backoff = [10, 30, 60]
    last_err = None
    for attempt in range(max_retries):
        try:
            pred.fetch_chain()
            break
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            if "rate" in err_msg or "too many" in err_msg or "429" in err_msg:
                wait = backoff[min(attempt, len(backoff) - 1)]
                time.sleep(wait)
            else:
                raise
    else:
        raise last_err

    pred.compute_greeks_vectorized()
    profile = pred.aggregate_exposures()
    by_strike = pred.by_strike.copy()

    S = profile['spot']
    vix = vol.fetch_vix_open()
    hv10 = vol.hv10()
    vanna_regime_label, vanna_adj = vol.classify_vanna_regime(vix)
    charm_regime = vol.bull_bear_regime()

    # Use ATM IV as 25-delta proxy
    atm_rows = pred.df[(pred.df['strike'] - S).abs() < 5]
    put_iv25 = float(atm_rows[atm_rows['type'] == 'put']['iv'].mean() or 0.20)
    call_iv25 = float(atm_rows[atm_rows['type'] == 'call']['iv'].mean() or 0.18)

    sr = se.compute_strikes(
        spot=S, vix=vix, hv10=hv10,
        put_iv25=put_iv25, call_iv25=call_iv25,
        vanna_adj=vanna_adj, charm_regime=charm_regime,
    )
    result = sig.evaluate(
        vix=vix, strike_result=sr, gvc_profile=profile,
        time_to_close_hours=3.0,
        condor_breakeven_width=sr['short_call'] - sr['short_put'],
        top_gex_strikes=by_strike.nlargest(5, 'strike_gex')['strike'].tolist(),
    )

    return {
        'profile': profile,
        'by_strike': by_strike,
        'strike_result': sr,
        'signal_result': result,
        'vix': vix,
        'hv10': hv10,
        'vanna_regime': vanna_regime_label,
        'charm_regime': charm_regime,
        'put_iv25': put_iv25,
        'call_iv25': call_iv25,
    }


def render_gvc_page():
    """Main render function for the GVC Signal Engine page."""

    st.markdown("""
    <style>
    .gvc-banner {
        font-family: 'Courier New', monospace;
        font-size: 18px;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .gvc-metric-card {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .gvc-metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .gvc-metric-value {
        font-size: 24px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("⚡ GVC Signal Engine")
    st.caption(
        "Gamma-Vanna-Charm Exposure Analysis • XSP 1DTE Iron Condor Signal • "
        "Dealer flow regime classification"
    )

    # ── Sidebar controls ──
    with st.sidebar:
        st.markdown("### GVC Settings")
        ticker = st.selectbox("Ticker", ["SPY", "XSP", "QQQ"], index=0,
                              help="SPY recommended — best options liquidity")
        strike_inc = st.select_slider("Strike Increment",
                                      options=[1.0, 2.0, 5.0, 10.0],
                                      value=5.0)

    # ── Refresh button ──
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        if st.button("🔄 Refresh GVC Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Run pipeline ──
    with st.spinner("Running GVC pipeline — fetching chain & computing exposures…"):
        try:
            data = _run_gvc_pipeline(ticker, strike_inc)
        except Exception as e:
            st.error(f"**Pipeline Error:** {e}")
            st.info("This may happen outside market hours or if the ticker lacks options data.")
            return

    profile = data['profile']
    by_strike = data['by_strike']
    sr = data['strike_result']
    result = data['signal_result']
    S = profile['spot']
    vix = data['vix']

    # ══════════════════════════════════════════════════
    # SIGNAL BANNER
    # ══════════════════════════════════════════════════
    sig_color = _signal_color(result['signal'])
    regime_emoji = _regime_emoji(result['regime'])

    st.markdown(f"""
    <div class="gvc-banner" style="background: linear-gradient(135deg, {sig_color}15, {sig_color}08);
         border-left: 4px solid {sig_color};">
        <span style="color: {sig_color}; font-size: 24px; font-weight: 800;">
            {result['signal']}
        </span>
        <span style="color: #aaa; margin-left: 16px;">
            {regime_emoji} {result['regime']}  •
            GEX: ${result['net_gex_B']:.2f}B  •
            Skew: {result['skew_label']}  •
            Vanna: {result['vanna_regime']}
        </span>
        <br>
        <span style="color: #ccc; font-size: 14px;">
            ➤ {result['action']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if result['flags']:
        for flag in result['flags']:
            st.warning(f"⚠️ {flag}")

    # ══════════════════════════════════════════════════
    # KEY METRICS ROW
    # ══════════════════════════════════════════════════
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spot", f"${S:.2f}")
    m2.metric("VIX", f"{vix:.2f}")
    m3.metric("HV10", f"{data['hv10']*100:.1f}%")
    m4.metric("Suitability", f"{result['suitability']}/100")
    m5.metric("Size", f"{result['size_multiplier']:.0%}")

    # ══════════════════════════════════════════════════
    # STRIKE RECOMMENDATION
    # ══════════════════════════════════════════════════
    st.markdown("### 🎯 Strike Recommendation")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Short Put", f"${result['short_put']:.0f}")
    s2.metric("Short Call", f"${result['short_call']:.0f}")
    s3.metric("Predicted Range",
              f"${result['predicted_range'][0]:.0f} – ${result['predicted_range'][1]:.0f}")
    s4.metric("Adjusted EM", f"${result['adjusted_em']:.2f}")

    # ── EM Breakdown ──
    with st.expander("📊 EM & Multiplier Breakdown"):
        e1, e2, e3 = st.columns(3)
        e1.metric("EM (IV-implied)", f"${sr['EM1']:.2f}")
        e2.metric("EM (Realized)", f"${sr['EM2']:.2f}")
        e3.metric("EM Base", f"${sr['EM_base']:.2f}")

        e4, e5, e6 = st.columns(3)
        e4.metric("Skew Ratio", f"{sr['skew_ratio']:.4f}", help=sr['skew_label'])
        e5.metric("Put Multiplier", f"{sr['put_multiplier']:.4f}")
        e6.metric("Call Multiplier", f"{sr['call_multiplier']:.4f}")

        e7, e8, e9 = st.columns(3)
        e7.metric("Vanna Adj", f"{sr['vanna_adj']:.2f}",
                  help=f"Regime: {data['vanna_regime']}")
        e8.metric("Charm Call Adj", f"{sr['charm_call_adj']:.2f}",
                  help=f"Regime: {data['charm_regime']}")
        e9.metric("Skew Adj", f"{sr['skew_adj']:.4f}")

    # ══════════════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════════════
    st.markdown("### 📈 Exposure Profiles")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "GEX by Strike", "Cumulative GEX", "Suitability Gauge"
    ])

    # ── GEX Bar Profile ──
    with chart_tab1:
        colors = ['#00cc55' if g > 0 else '#ff4444'
                  for g in by_strike['strike_gex']]
        gex_fig = go.Figure()
        gex_fig.add_trace(go.Bar(
            x=by_strike['strike'],
            y=by_strike['strike_gex'] / 1e9,
            marker_color=colors,
            name='GEX',
            hovertemplate='Strike: %{x}<br>GEX: $%{y:.3f}B<extra></extra>'
        ))
        gex_fig.add_vline(x=S, line_dash='dash', line_color='yellow',
                          annotation_text=f"Spot {S:.2f}")
        gex_fig.add_vline(x=profile['pin_strike'], line_dash='dot',
                          line_color='cyan',
                          annotation_text=f"Pin {profile['pin_strike']:.0f}")
        for w in profile['gamma_walls']:
            gex_fig.add_vline(x=w, line_dash='longdash',
                              line_color='orange', opacity=0.5)
        # Mark short strikes
        gex_fig.add_vline(x=result['short_put'], line_dash='solid',
                          line_color='#ff66ff', opacity=0.7,
                          annotation_text=f"SP {result['short_put']:.0f}")
        gex_fig.add_vline(x=result['short_call'], line_dash='solid',
                          line_color='#ff66ff', opacity=0.7,
                          annotation_text=f"SC {result['short_call']:.0f}")
        gex_fig.update_layout(
            title='GEX by Strike ($B)',
            template='plotly_dark',
            xaxis_title='Strike', yaxis_title='GEX ($B)',
            height=450,
            xaxis_range=[S - 50, S + 50],
        )
        st.plotly_chart(gex_fig, use_container_width=True)

    # ── Cumulative GEX ──
    with chart_tab2:
        cum_fig = go.Figure()
        cum_fig.add_trace(go.Scatter(
            x=by_strike['strike'],
            y=by_strike['cum_gex'] / 1e9,
            line=dict(color='#00aaff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,170,255,0.1)',
            name='Cumulative GEX',
            hovertemplate='Strike: %{x}<br>Cum GEX: $%{y:.3f}B<extra></extra>'
        ))
        cum_fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
        cum_fig.add_vline(x=profile['zero_gamma_strike'], line_dash='dot',
                          line_color='red',
                          annotation_text=f"Zero-Gamma {profile['zero_gamma_strike']:.0f}")
        cum_fig.update_layout(
            title='Cumulative GEX ($B)',
            template='plotly_dark',
            xaxis_title='Strike', yaxis_title='Cum GEX ($B)',
            height=450,
            xaxis_range=[S - 50, S + 50],
        )
        st.plotly_chart(cum_fig, use_container_width=True)

    # ── Suitability Gauge ──
    with chart_tab3:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['suitability'],
            title={'text': f"Suitability — {result['signal']}",
                   'font': {'size': 20, 'color': 'white'}},
            number={'font': {'size': 48, 'color': sig_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': sig_color},
                'bgcolor': '#1a1a2e',
                'steps': [
                    {'range': [0, 35], 'color': '#1a0000'},
                    {'range': [35, 60], 'color': '#1a1000'},
                    {'range': [60, 85], 'color': '#001a00'},
                    {'range': [85, 100], 'color': '#003300'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 2},
                    'thickness': 0.8,
                    'value': result['suitability'],
                },
            }
        ))
        gauge_fig.update_layout(
            template='plotly_dark',
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    # ══════════════════════════════════════════════════
    # PROFILE DETAILS
    # ══════════════════════════════════════════════════
    with st.expander("🔬 Full GVC Profile Details"):
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Exposure Metrics**")
            st.json({
                'Net GEX': f"${profile['net_gex']/1e9:.4f}B",
                'GEX Strength': f"{profile['gex_strength']:.4f}",
                'Net VEX': f"{profile['net_vex']:.2f}",
                'Net CEX': f"{profile['net_cex']:.2f}",
                'Vanna Skew': f"{profile['vanna_skew']:.2f}",
            })
        with p2:
            st.markdown("**Key Levels**")
            st.json({
                'Spot': f"${S:.2f}",
                'Zero-Gamma': f"${profile['zero_gamma_strike']:.0f}",
                'Pin Strike': f"${profile['pin_strike']:.0f}",
                'Gamma Walls': [f"${w:.0f}" for w in profile['gamma_walls'][:8]],
            })

    st.markdown("---")
    st.caption(
        "⚠️ yfinance OI is end-of-day — GEX values reflect prior close. "
        "Not financial advice. For research and development purposes only."
    )
