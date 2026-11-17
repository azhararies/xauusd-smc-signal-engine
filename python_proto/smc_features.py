"""
smc_features.py - Smart Money Concepts Feature Engineering
Tính các đặc trưng cấu trúc thị trường từ OHLCV:
  - Swing High / Swing Low (pivot points)
  - Market Structure: HH/HL (Uptrend) vs LH/LL (Downtrend)
  - Order Block: Nến OB trước cú bùng phá mạnh
  - Fair Value Gap (FVG / Imbalance): Khoảng trống 3 nến chưa được fill
  - Distance từ giá hiện tại đến OB/FVG gần nhất
"""
import pandas as pd
import numpy as np


def find_swing_points(df, lookback=5):
    """
    Xác định Swing High và Swing Low.
    Swing High: nến i có High cao nhất trong [i-lb, i+lb]
    Swing Low:  nến i có Low  thấp nhất trong [i-lb, i+lb]
    """
    highs = df['High'].values
    lows  = df['Low'].values
    n     = len(df)
    swing_highs = np.zeros(n, dtype=float)
    swing_lows  = np.zeros(n, dtype=float)

    for i in range(lookback, n - lookback):
        left_h  = highs[i - lookback:i]
        right_h = highs[i + 1:i + lookback + 1]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            swing_highs[i] = highs[i]

        left_l  = lows[i - lookback:i]
        right_l = lows[i + 1:i + lookback + 1]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            swing_lows[i] = lows[i]

    df = df.copy()
    df['SwingHigh'] = swing_highs
    df['SwingLow']  = swing_lows
    return df


def get_market_structure(df):
    """
    Xác định cấu trúc thị trường dựa trên chuỗi Swing.
    1 = Bullish (HH/HL), -1 = Bearish (LH/LL), 0 = Neutral
    """
    structure = np.zeros(len(df))
    pivot_highs = df[df['SwingHigh'] > 0]['SwingHigh']
    pivot_lows  = df[df['SwingLow']  > 0]['SwingLow']

    # Tính slope của chuỗi swing highs và swing lows (rolling 2 giá trị)
    ph_diff = pivot_highs.diff()
    pl_diff = pivot_lows.diff()

    for i in range(len(df)):
        ts = df.index[i]
        # Lấy 2 swing gần nhất trước ts
        recent_ph = ph_diff[ph_diff.index < ts].tail(1)
        recent_pl = pl_diff[pl_diff.index < ts].tail(1)

        ph_up = (recent_ph > 0).any()   # Higher High
        pl_up = (recent_pl > 0).any()   # Higher Low
        ph_dn = (recent_ph < 0).any()   # Lower High
        pl_dn = (recent_pl < 0).any()   # Lower Low

        if ph_up and pl_up:
            structure[i] = 1   # Bullish
        elif ph_dn and pl_dn:
            structure[i] = -1  # Bearish

    df = df.copy()
    df['MS_Structure'] = structure
    return df


def find_order_blocks(df, ob_lookback=3, min_move_atr=1.5):
    """
    Order Block: Nến bearish cuối cùng trước khi giá bùng tăng mạnh (Bullish OB)
                 Nến bullish cuối cùng trước khi giá giảm mạnh  (Bearish OB)
    Điều kiện bùng phá: Nến tiếp theo có Range >= min_move_atr × ATR_14
    """
    df = df.copy()
    atr14 = (df['High'] - df['Low']).rolling(14).mean()

    bull_ob_high = np.nan * np.ones(len(df))
    bull_ob_low  = np.nan * np.ones(len(df))
    bear_ob_high = np.nan * np.ones(len(df))
    bear_ob_low  = np.nan * np.ones(len(df))

    for i in range(1, len(df) - 1):
        next_range = df['High'].iloc[i] - df['Low'].iloc[i]
        threshold  = min_move_atr * (atr14.iloc[i] if not np.isnan(atr14.iloc[i]) else 1)

        # Bullish OB: Nến trước là bearish, nến tiếp theo tăng mạnh
        is_bullish_impulse = (df['Close'].iloc[i] > df['Open'].iloc[i] and next_range >= threshold)
        if is_bullish_impulse and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
            bull_ob_high[i-1] = df['Open'].iloc[i-1]
            bull_ob_low[i-1]  = df['Low'].iloc[i-1]

        # Bearish OB: Nến trước là bullish, nến tiếp theo giảm mạnh
        is_bearish_impulse = (df['Close'].iloc[i] < df['Open'].iloc[i] and next_range >= threshold)
        if is_bearish_impulse and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:
            bear_ob_high[i-1] = df['High'].iloc[i-1]
            bear_ob_low[i-1]  = df['Close'].iloc[i-1]

    df['BullOB_High'] = bull_ob_high
    df['BullOB_Low']  = bull_ob_low
    df['BearOB_High'] = bear_ob_high
    df['BearOB_Low']  = bear_ob_low
    return df


def find_fair_value_gaps(df):
    """
    Fair Value Gap (FVG / Imbalance):
    Bullish FVG: High[i-1] < Low[i+1] (gap lên trên)
    Bearish FVG: Low[i-1]  > High[i+1] (gap xuống dưới)
    """
    df = df.copy()
    n = len(df)
    bull_fvg = np.zeros(n)
    bear_fvg = np.zeros(n)
    bull_fvg_size = np.zeros(n)
    bear_fvg_size = np.zeros(n)

    for i in range(1, n - 1):
        gap_up   = df['Low'].iloc[i+1]  - df['High'].iloc[i-1]
        gap_down = df['Low'].iloc[i-1]  - df['High'].iloc[i+1]
        if gap_up > 0:
            bull_fvg[i]      = 1
            bull_fvg_size[i] = gap_up
        if gap_down > 0:
            bear_fvg[i]      = 1
            bear_fvg_size[i] = gap_down

    df['BullFVG']      = bull_fvg
    df['BearFVG']      = bear_fvg
    df['BullFVG_Size'] = bull_fvg_size
    df['BearFVG_Size'] = bear_fvg_size
    return df


def add_distance_features(df, lookback=50):
    """
    Tính khoảng cách từ giá hiện tại đến:
    - Swing High/Low gần nhất (normalized bởi ATR)
    - Order Block gần nhất
    - FVG gần nhất
    """
    df = df.copy()
    atr14 = (df['High'] - df['Low']).rolling(14).mean().values
    close = df['Close'].values

    dist_sh    = np.full(len(df), np.nan)  # Distance to Swing High
    dist_sl    = np.full(len(df), np.nan)  # Distance to Swing Low
    dist_bull_ob = np.full(len(df), np.nan)
    dist_bear_ob = np.full(len(df), np.nan)
    near_fvg   = np.zeros(len(df))

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        atr    = atr14[i] if not np.isnan(atr14[i]) else 1

        # Swing High/Low distance
        sh_vals = window['SwingHigh'][window['SwingHigh'] > 0]
        sl_vals = window['SwingLow'][window['SwingLow'] > 0]
        if not sh_vals.empty:
            dist_sh[i] = (sh_vals.iloc[-1] - close[i]) / atr
        if not sl_vals.empty:
            dist_sl[i] = (close[i] - sl_vals.iloc[-1]) / atr

        # Order Block distance
        bull_obs = window['BullOB_High'].dropna()
        bear_obs = window['BearOB_Low'].dropna()
        if not bull_obs.empty:
            dist_bull_ob[i] = (close[i] - bull_obs.iloc[-1]) / atr
        if not bear_obs.empty:
            dist_bear_ob[i] = (bear_obs.iloc[-1] - close[i]) / atr

        # FVG nearby (trong 10 nến gần nhất)
        recent = window.tail(10)
        if recent['BullFVG'].any() or recent['BearFVG'].any():
            near_fvg[i] = 1

    df['Dist_SwingHigh']  = dist_sh
    df['Dist_SwingLow']   = dist_sl
    df['Dist_BullOB']     = dist_bull_ob
    df['Dist_BearOB']     = dist_bear_ob
    df['Near_FVG']        = near_fvg
    return df


def calculate_all_smc(df, swing_lookback=5, ob_min_atr=1.5, dist_lookback=50):
    """Pipeline tổng hợp tất cả SMC features"""
    print("  [SMC] Swing Points...")
    df = find_swing_points(df, lookback=swing_lookback)
    print("  [SMC] Market Structure...")
    df = get_market_structure(df)
    print("  [SMC] Order Blocks...")
    df = find_order_blocks(df, min_move_atr=ob_min_atr)
    print("  [SMC] Fair Value Gaps...")
    df = find_fair_value_gaps(df)
    print("  [SMC] Distance Features...")
    df = add_distance_features(df, lookback=dist_lookback)
    return df


if __name__ == "__main__":
    # Quick test
    import pandas as pd, numpy as np
    from resample import load_m1, resample_ohlcv
    df_m1 = load_m1()
    df_m5 = resample_ohlcv(df_m1, '5min').tail(5000)
    print(f"Test SMC trên {len(df_m5)} nến M5...")
    df_smc = calculate_all_smc(df_m5)
    smc_cols = ['SwingHigh','SwingLow','MS_Structure','BullFVG','BearFVG','Dist_SwingHigh','Dist_SwingLow','Near_FVG']
    print(df_smc[smc_cols].tail(5).to_string())
    print(f"[+] Swing Highs tìm được: {(df_smc['SwingHigh']>0).sum()}")
    print(f"[+] Swing Lows  tìm được: {(df_smc['SwingLow']>0).sum()}")
    print(f"[+] Bullish FVG: {df_smc['BullFVG'].sum()} | Bearish FVG: {df_smc['BearFVG'].sum()}")
    print(f"[+] Bullish OB: {(~df_smc['BullOB_High'].isna()).sum()} | Bearish OB: {(~df_smc['BearOB_High'].isna()).sum()}")
