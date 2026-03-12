"""
walk_forward.py - Walk-Forward Backtest 12 Tháng
Mỗi lần lặp: Train trên 6 tháng trước, Test trên 1 tháng tiếp theo.
Chạy 12 lần → Biết tháng nào Model hoạt động tốt/xấu → EV có ổn định không?

Đây là cách test chống overfitting đúng chuẩn, tránh look-ahead bias.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from smc_features import calculate_all_smc

DATA_2024  = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
DATA_2025  = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"

CONFIDENCE    = 0.60
MAX_BARS_HOLD = 30       # Max 30 nến M5 = 2.5h
RISK_PCT      = 0.01
STARTING_BAL  = 100_000
TRAIN_MONTHS  = 6        # Train trên 6 tháng trước
SL_MIN        = 2.0
SL_MAX        = 5.0
RR            = 1.5

XGB_FEATURES  = [
    'Body_Pct','Upper_Wick','Lower_Wick','BodyATR','RangeATR',
    'RSI_14','ROC_5','ROC_20','Trend_Up','Vol_Ratio',
    'MS_Structure','Near_FVG','BullFVG','BearFVG',
    'Dist_SwingHigh','Dist_SwingLow','Dist_BullOB','Dist_BearOB',
]

def load_full_data():
    cols = ['Date','Time','Open','High','Low','Close','Volume']
    df24 = pd.read_csv(DATA_2024, sep=',', names=cols)
    df25 = pd.read_csv(DATA_2025, sep=',', names=cols)
    df = pd.concat([df24, df25], ignore_index=True)
    df['Datetime'] = pd.to_datetime(df['Date']+' '+df['Time'], format='%Y.%m.%d %H:%M')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date','Time'], inplace=True)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    # Resample M5
    df = df.resample('5min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    return df

def add_features(df):
    df = df.copy()
    df['Body']       = df['Close'] - df['Open']
    df['Body_Pct']   = (df['Body'] / df['Open']) * 100
    df['Range']      = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['ATR_14']     = df['Range'].rolling(14).mean()
    df['BodyATR']    = df['Body'].abs() / df['ATR_14']
    df['RangeATR']   = df['Range'] / df['ATR_14']
    delta = df['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14']     = 100 - (100 / (1 + g / l))
    df['ROC_5']      = df['Close'].pct_change(5) * 100
    df['ROC_20']     = df['Close'].pct_change(20) * 100
    df['Vol_Ratio']  = df['Volume'] / df['Volume'].rolling(20).mean()
    df['SMA_50']     = df['Close'].rolling(50).mean()
    df['SMA_200']    = df['Close'].rolling(200).mean()
    df['Trend_Up']   = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Hour']       = df.index.hour
    df['Is_Active']  = np.where((df['Hour'] >= 7) & (df['Hour'] < 21), 1, 0)
    df['Target']     = np.where((df['Close'].shift(-1) - df['Open'].shift(-1)) > 0, 1, 0)
    df.dropna(inplace=True)
    return df

def get_structural_sl(window_df, direction):
    """SL = Swing Low/High gần nhất trong cửa sổ"""
    if direction == 'LONG':
        sl_vals = window_df['SwingLow'][window_df['SwingLow'] > 0]
        if not sl_vals.empty:
            return sl_vals.iloc[-1] * 0.9999
        return window_df['Low'].min() * 0.9999
    else:
        sl_vals = window_df['SwingHigh'][window_df['SwingHigh'] > 0]
        if not sl_vals.empty:
            return sl_vals.iloc[-1] * 1.0001
        return window_df['High'].max() * 1.0001

def backtest_month(test_df):
    """Chạy backtest trên 1 tháng data"""
    wins = losses = 0
    i = 0
    test_df = test_df.reset_index(drop=True)
    while i < len(test_df) - MAX_BARS_HOLD:
        row = test_df.iloc[i]
        if row.get('Is_Active', 1) == 0:
            i += 1; continue

        if row['Prob_Up'] >= CONFIDENCE and row['Trend_Up'] == 1:
            direction = 'LONG'
        elif row['Prob_Up'] <= 1 - CONFIDENCE and row['Trend_Up'] == -1:
            direction = 'SHORT'
        else:
            i += 1; continue

        entry = row['Open']
        window = test_df.iloc[max(0, i-20):i]
        sl     = get_structural_sl(window, direction)
        sl_dist = abs(entry - sl)

        if sl_dist < SL_MIN or sl_dist > SL_MAX:
            i += 1; continue

        tp = entry + sl_dist * RR if direction == 'LONG' else entry - sl_dist * RR
        outcome   = 'OPEN'
        bars_held = 0
        for j in range(i+1, min(i+1+MAX_BARS_HOLD, len(test_df))):
            bar = test_df.iloc[j]; bars_held = j - i
            if direction == 'LONG':
                if bar['Low']  <= sl: outcome = 'LOSS'; break
                if bar['High'] >= tp: outcome = 'WIN';  break
            else:
                if bar['High'] >= sl: outcome = 'LOSS'; break
                if bar['Low']  <= tp: outcome = 'WIN';  break
        if outcome == 'OPEN': outcome = 'LOSS'
        if outcome == 'WIN': wins += 1
        else: losses += 1
        i += (bars_held or 1) + 1

    total = wins + losses
    if total == 0:
        return {'wins': 0, 'losses': 0, 'total': 0, 'winrate': 0, 'ev': 0}
    wr = wins / total
    ev = wr * RR - (1 - wr)
    return {'wins': wins, 'losses': losses, 'total': total, 'winrate': wr, 'ev': ev}

def run_walk_forward():
    print("[*] Load data...")
    df_raw = load_full_data()
    print(f"[*] Feature Engineering...")
    df = add_features(df_raw)
    print(f"[*] SMC Features (chạy 1 lần trên toàn bộ data)...")
    df = calculate_all_smc(df, swing_lookback=5, dist_lookback=50)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=XGB_FEATURES + ['Target'])

    # Xác định các tháng test
    months = df.index.to_period('M').unique()
    # Chỉ test từ tháng thứ TRAIN_MONTHS+1 trở đi
    test_months  = months[TRAIN_MONTHS:]

    print(f"\n{'='*70}")
    print(f" WALK-FORWARD TEST ({TRAIN_MONTHS}M Train / 1M Test) — XGBoost + SMC")
    print(f"{'='*70}")
    print(f"{'Tháng':^12}{'Lệnh':>8}{'Win':>6}{'Loss':>6}{'WR%':>8}{'EV/R':>10}{'Kết luận':>14}")
    print('-'*70)

    monthly_results = []
    for test_period in test_months:
        # Train window: TRAIN_MONTHS tháng trước test_period
        train_start = (test_period - TRAIN_MONTHS).start_time
        train_end   = test_period.start_time
        test_start  = test_period.start_time
        test_end    = test_period.end_time

        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df  = df[(df.index >= test_start)  & (df.index <= test_end)]

        if len(train_df) < 500 or len(test_df) < 50:
            continue

        # Train XGBoost
        X_train = train_df[XGB_FEATURES]
        y_train = train_df['Target']
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict trên test month
        test_copy = test_df.copy()
        test_copy['Prob_Up'] = model.predict_proba(test_df[XGB_FEATURES])[:, 1]

        # Backtest
        result = backtest_month(test_copy)
        result['month'] = str(test_period)
        monthly_results.append(result)

        ev_sign = "✅ EV+" if result['ev'] > 0 else "❌ EV-"
        print(f"  {str(test_period):^12}{result['total']:>8}{result['wins']:>6}"
              f"{result['losses']:>6}{result['winrate']*100:>7.1f}%{result['ev']:>+10.3f}R  {ev_sign}")

    # Tổng kết
    if not monthly_results:
        print("Không đủ data để Walk-Forward!"); return

    total_trades  = sum(r['total'] for r in monthly_results)
    total_wins    = sum(r['wins'] for r in monthly_results)
    total_losses  = sum(r['losses'] for r in monthly_results)
    pos_ev_months = sum(1 for r in monthly_results if r['ev'] > 0)
    all_ev        = [r['ev'] for r in monthly_results if r['total'] > 0]
    avg_ev        = np.mean(all_ev) if all_ev else 0
    med_ev        = np.median(all_ev) if all_ev else 0

    wr_total = total_wins / total_trades if total_trades > 0 else 0
    ev_total = wr_total * RR - (1 - wr_total)

    print(f"{'='*70}")
    print(f"  TỔNG KẾT WALK-FORWARD ({len(monthly_results)} tháng test)")
    print(f"{'='*70}")
    print(f"  Tổng lệnh    : {total_trades} (Win:{total_wins} Loss:{total_losses})")
    print(f"  WR Gộp       : {wr_total*100:.1f}%")
    print(f"  EV Gộp       : {ev_total:+.3f}R  {'✅ EV DƯƠNG!' if ev_total > 0 else '❌ EV âm'}")
    print(f"  EV Avg/tháng : {avg_ev:+.3f}R | Median: {med_ev:+.3f}R")
    print(f"  Tháng EV+    : {pos_ev_months}/{len(monthly_results)} tháng")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_walk_forward()
