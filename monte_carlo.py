import numpy as np
import pandas as pd
import math
from datetime import datetime
from gvc_engine import StrikeEngine, SkewProbabilityEngine, SignalLayer

def simulate():
    np.random.seed(42)
    ITERATIONS = 10000

    # Fixed parameters
    SPOT_PRICE = 500.0
    TIME_TO_CLOSE_HOURS = 24.0 # 1 day
    CONDOR_BREAKEVEN_WIDTH = 1.0 # 1 pt wings
    WING_WIDTH = 1.0
    MIN_CREDIT = 0.20

    print(f"Running Monte Carlo Simulation with {ITERATIONS} iterations...")

    # Generate Synthetic Market Environment
    # RV10: Lognormal centered around 14%
    rv10 = np.random.lognormal(mean=np.log(14), sigma=0.4, size=ITERATIONS)
    rv10 = np.clip(rv10, 5, 80) # Bound to realistic values

    # ATM IV: Correlated to RV10, with a Variance Risk Premium
    vrp = np.random.normal(loc=1.5, scale=2.0, size=ITERATIONS)
    atm_iv = rv10 + vrp
    atm_iv = np.clip(atm_iv, 5, 100) # Bound

    # Skew Ratio (Put IV / Call IV): typically > 1 (bearish skew)
    skew_ratios = np.random.normal(loc=1.15, scale=0.15, size=ITERATIONS)
    skew_ratios = np.clip(skew_ratios, 0.8, 1.8)

    # Derive Put/Call IVs
    # total_iv = put_iv + call_iv.  Assume (put_iv + call_iv)/2 is roughly atm_iv * 1.05 (OTM smile)
    # This is a simplification. Let's just solve: put_iv / call_iv = skew_ratio, (put_iv + call_iv)/2 = atm_iv
    # put_iv = skew_ratio * call_iv
    # (skew_ratio * call_iv + call_iv) / 2 = atm_iv
    # call_iv = 2 * atm_iv / (skew_ratio + 1)
    call_ivs = 2 * atm_iv / (skew_ratios + 1)
    put_ivs = skew_ratios * call_ivs

    # VIX (proxy for ATM IV)
    vixs = atm_iv

    # Generate GVC profiles
    # Net GEX: Normally positive, occasional fat tail negative
    net_gexs = np.random.normal(loc=2e9, scale=3e9, size=ITERATIONS)
    net_vexs = np.random.normal(loc=0, scale=2e6, size=ITERATIONS)
    net_cexs = np.random.normal(loc=0, scale=2e5, size=ITERATIONS)

    # Vanna/Charm regimes (Randomized based on VRP and Skew roughly)
    vanna_adjs = np.where(vrp > 3, 1.12, np.where(vrp < -1, 0.92, 1.00))
    charm_regimes = np.where(net_gexs > 3e9, 'bull', np.where(net_gexs < -1e9, 'bear', 'neutral'))

    # Generate Next Day Settlement Price using GBM
    # We use actual RV10 as the volatility for the step
    # S_t = S_0 * exp((r - 0.5*sigma^2)*t + sigma*sqrt(t)*Z)
    t = 1 / 252.0
    r = 0.05
    z = np.random.standard_normal(ITERATIONS)
    sigma = rv10 / 100.0
    
    # Calculate geometric brownian motion step
    drift = (r - 0.5 * sigma**2) * t
    shock = sigma * np.sqrt(t) * z
    settle_prices = SPOT_PRICE * np.exp(drift + shock)

    # Instantiating engines
    strike_engine = StrikeEngine(strike_increment=5.0) # wait app uses 1.0 increments usually for spy, let's use 1.0 for spy proxy
    strike_engine.strike_increment = 1.0
    skew_engine = SkewProbabilityEngine()
    signal_layer = SignalLayer()

    vrp_results = []
    gvc_results = []

    for i in range(ITERATIONS):
        s0 = SPOT_PRICE
        rv = rv10[i]
        iv = atm_iv[i]
        p_iv = put_ivs[i]
        c_iv = call_ivs[i]
        gex = net_gexs[i]
        vex = net_vexs[i]
        cex = net_cexs[i]
        vanna = vanna_adjs[i]
        charm = charm_regimes[i]
        settle = settle_prices[i]
        
        # 1. VRP Scanner Logic
        v_vrp = iv - rv
        adj_iv_vrp = max(iv - v_vrp, 5) if v_vrp > 0 else iv
        
        em1_pct = (iv / 100) * np.sqrt(t) * 100
        em1_dol = s0 * (em1_pct / 100)
        em2_pct = (adj_iv_vrp / 100) * np.sqrt(t) * 100
        em2_dol = s0 * (em2_pct / 100)
        
        adj_em_dol = (em1_dol + em2_dol) / 2
        
        vrp_short_put = round(s0 - adj_em_dol * 0.98)
        vrp_short_call = round(s0 + adj_em_dol * 0.98)
        
        vrp_est_credit = 0.15 + max(0, v_vrp / 15) * 0.35
        vrp_skip = v_vrp > 5 # Load day
        
        if not vrp_skip:
            put_loss = max(min(vrp_short_put - settle, WING_WIDTH), 0)
            call_loss = max(min(settle - vrp_short_call, WING_WIDTH), 0)
            pnl = vrp_est_credit - put_loss - call_loss
            pnl = max(pnl, -(WING_WIDTH - vrp_est_credit))
            vrp_results.append(pnl)

        # 2. GVC Signal Engine Logic
        gvc_prof = {
            'spot': s0,
            'net_gex': gex,
            'net_vex': vex,
            'net_cex': cex
        }
        
        # Get base EM info
        em_info = strike_engine.compute_expected_move(s0, iv, rv / 100.0)
        em_base = em_info['EM_base']
        
        # Skew/Asymmetric Profile
        skew_profile = skew_engine.full_asymmetric_profile(em_base, p_iv, c_iv)
        
        # Strike computation
        strike_result = strike_engine.compute_strikes(
            spot=s0,
            vix=iv,
            hv10=rv / 100.0,
            put_iv25=p_iv,
            call_iv25=c_iv,
            vanna_adj=vanna,
            charm_regime=charm,
            asymmetric_em=skew_profile
        )
        
        # Signal evaluation
        signal_result = signal_layer.evaluate(
            vix=iv,
            strike_result=strike_result,
            gvc_profile=gvc_prof,
            time_to_close_hours=TIME_TO_CLOSE_HOURS,
            condor_breakeven_width=CONDOR_BREAKEVEN_WIDTH,
            skew_profile=skew_profile
        )
        
        gvc_signal = signal_result['signal']
        g_short_put = signal_result['short_put']
        g_short_call = signal_result['short_call']
        
        if gvc_signal in ['STRONG', 'MODERATE', 'WEAK', 'CAUTION', 'HIGH_RISK']:
            # Assume credit roughly correlates to signal strength/VRP
            # For backtest parity, we give them same credit logic but modified by size_multiplier
            # Or we can just calculate standard credit. Let's use the standard credit for apples to apples
            gvc_credit = 0.15 + max(0, v_vrp / 15) * 0.35
            
            put_loss = max(min(g_short_put - settle, WING_WIDTH), 0)
            call_loss = max(min(settle - g_short_call, WING_WIDTH), 0)
            pnl = gvc_credit - put_loss - call_loss
            pnl = max(pnl, -(WING_WIDTH - gvc_credit))
            gvc_results.append((pnl, gvc_signal))


    # Compile Results
    vrp_wins = sum(1 for p in vrp_results if p > 0)
    vrp_total = len(vrp_results)
    vrp_winrate = vrp_wins / vrp_total * 100 if vrp_total > 0 else 0
    vrp_avg_pnl = sum(vrp_results) / vrp_total * 100 if vrp_total > 0 else 0

    gvc_pnl_all = [x[0] for x in gvc_results]
    gvc_wins = sum(1 for p in gvc_pnl_all if p > 0)
    gvc_total = len(gvc_pnl_all)
    gvc_winrate = gvc_wins / gvc_total * 100 if gvc_total > 0 else 0
    gvc_avg_pnl = sum(gvc_pnl_all) / gvc_total * 100 if gvc_total > 0 else 0

    gvc_strong = [x[0] for x in gvc_results if x[1] == 'STRONG']
    strong_wins = sum(1 for p in gvc_strong if p > 0)
    strong_total = len(gvc_strong)
    strong_winrate = strong_wins / strong_total * 100 if strong_total > 0 else 0
    strong_avg_pnl = sum(gvc_strong) / strong_total * 100 if strong_total > 0 else 0

    print("\n" + "="*50)
    print("MONTE CARLO SIMULATION RESULTS (10,000 Days)")
    print("="*50)
    
    print("\n[ VRP SCANNER (Classic) ]")
    print(f"Trades Taken: {vrp_total}")
    print(f"Win Rate:     {vrp_winrate:.2f}%")
    print(f"Avg P&L/Trade: ${vrp_avg_pnl:.2f}")

    print("\n[ GVC SIGNAL ENGINE (All Signals) ]")
    print(f"Trades Taken: {gvc_total}")
    print(f"Win Rate:     {gvc_winrate:.2f}%")
    print(f"Avg P&L/Trade: ${gvc_avg_pnl:.2f}")

    print("\n[ GVC SIGNAL ENGINE ('STRONG' Signals Only) ]")
    print(f"Trades Taken: {strong_total}")
    print(f"Win Rate:     {strong_winrate:.2f}%")
    print(f"Avg P&L/Trade: ${strong_avg_pnl:.2f}")

if __name__ == "__main__":
    simulate()
