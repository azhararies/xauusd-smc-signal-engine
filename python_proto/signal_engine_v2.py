"""
signal_engine_v2.py - XGBoost + SMC Features Signal Engine
Nâng cấp từ KNN (signal_engine.py) lên:
  1. SMC Features: Swing Structure, Order Block, FVG, Market Structure
  2. XGBoost Classifier (mạnh hơn KNN, có feature importance)
  3. Structural SL/TP: SL = Swing Low/High gần nhất (không dùng ATR × N)
  4. HMM Regime Filter: Chỉ trade khi đúng regime (Trending / Ranging)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import xgboost as xgb
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from smc_features import calculate_all_smc

DATA_2024  = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
DATA_2025  = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"
TIMEFRAME  = 'M5'

CONFIDENCE    = 0.60     # XGBoost predict_proba threshold
MAX_BARS_HOLD = 30       # Tối đa 30 nến M5 = 2.5h giữ lệnh
RISK_PCT      = 0.01
STARTING_BAL  = 100_000

def load_resample():
    cols = ['Date','Time','Open','High','Low','Close','Volume']
    df24 = pd.read_csv(DATA_2024, sep=',', names=cols)
    df25 = pd.read_csv(DATA_2025, sep=',', names=cols)
    df = pd.concat([df24, df25], ignore_index=True)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date','Time'], inplace=True)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    df = df.resample('5min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    return df

def add_base_features(df):
    df = df.copy()
    df['Body']       = df['Close'] - df['Open']
    df['Body_Pct']   = (df['Body'] / df['Open']) * 100
    df['Range']      = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['ATR_14']     = df['Range'].rolling(14).mean()
    df['BodyATR']    = df['Body'].abs() / df['ATR_14']
    df['RangeATR']   = df['Range'] / df['ATR_14']
    delta            = df['Close'].diff()
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
    df['Is_Active']  = np.where(((df['Hour'] >= 7) & (df['Hour'] < 21)), 1, 0)
    # Target
    df['Next_Body']  = df['Close'].shift(-1) - df['Open'].shift(-1)
    df['Target']     = np.where(df['Next_Body'] > 0, 1, 0)
    return df

XGB_FEATURES  = [
    'Body_Pct', 'Upper_Wick', 'Lower_Wick', 'BodyATR', 'RangeATR',
    'RSI_14', 'ROC_5', 'ROC_20', 'Trend_Up', 'Vol_Ratio',
    # SMC Features
    'MS_Structure', 'Near_FVG', 'BullFVG', 'BearFVG',
    'Dist_SwingHigh', 'Dist_SwingLow', 'Dist_BullOB', 'Dist_BearOB',
]

def train_xgboost(df):
    df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=XGB_FEATURES + ['Target'])
    df2.reset_index(drop=True, inplace=True)

    split = int(len(df2) * 0.8)
    X_train = df2[XGB_FEATURES].iloc[:split]
    y_train = df2['Target'].iloc[:split]
    X_test  = df2[XGB_FEATURES].iloc[split:]
    y_test  = df2['Target'].iloc[split:]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,              # Shallow tree chống overfit
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,      # Tránh overfit trên sample nhỏ
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)
    acc = (pred_test == y_test).mean()
    print(f"[+] XGBoost Direction Accuracy (50% threshold): {acc*100:.2f}%")

    # Feature Importance top 10
    importance = dict(zip(XGB_FEATURES, model.feature_importances_))
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("[+] Top 10 Features:")
    for feat, imp in top10:
        print(f"    {feat:25s}: {imp:.4f}")

    test_df = df2.iloc[split:].copy()
    test_df['Prob_Up'] = prob_test
    return model, test_df

def get_structural_sl(test_df, sig_idx, direction, lookback=20):
    """SL = Swing Low/High gần nhất (đã tính trong SMC features)"""
    start = max(0, sig_idx - lookback)
    window = test_df.iloc[start:sig_idx]

    if direction == 'LONG':
        sl_candidates = window['SwingLow'][window['SwingLow'] > 0]
        if not sl_candidates.empty:
            return sl_candidates.iloc[-1] * 0.9999  # Ngay dưới swing low
        return window['Low'].min() * 0.9999
    else:
        sl_candidates = window['SwingHigh'][window['SwingHigh'] > 0]
        if not sl_candidates.empty:
            return sl_candidates.iloc[-1] * 1.0001
        return window['High'].max() * 1.0001

def run_backtest(test_df):
    equity  = STARTING_BAL
    hwm     = STARTING_BAL
    floor   = STARTING_BAL * 0.9
    risk_usd = STARTING_BAL * RISK_PCT
    wins = losses = 0
    equity_curve = [equity]
    results = []

    i = 0
    test_df = test_df.reset_index(drop=True)
    while i < len(test_df) - MAX_BARS_HOLD:
        row = test_df.iloc[i]

        # Filter: Active session + Confidence + Trend
        if row['Is_Active'] == 0:
            i += 1; continue

        if row['Prob_Up'] >= CONFIDENCE and row['Trend_Up'] == 1:
            direction = 'LONG'
        elif row['Prob_Up'] <= 1 - CONFIDENCE and row['Trend_Up'] == -1:
            direction = 'SHORT'
        else:
            i += 1; continue

        # Structural SL/TP
        entry = row['Open']
        sl    = get_structural_sl(test_df, i, direction, lookback=20)
        sl_dist = abs(entry - sl)

        # Lọc SL quá nhỏ (< 2 pip) hoặc quá lớn (> 50 pip = 5.0 price)
        if sl_dist < 2.0 or sl_dist > 5.0:
            i += 1; continue

        rr = 1.5
        tp = entry + sl_dist * rr if direction == 'LONG' else entry - sl_dist * rr

        outcome   = 'OPEN'
        bars_held = 0
        for j in range(i + 1, min(i + 1 + MAX_BARS_HOLD, len(test_df))):
            bar = test_df.iloc[j]
            bars_held = j - i
            if direction == 'LONG':
                if bar['Low']  <= sl: outcome = 'LOSS'; break
                if bar['High'] >= tp: outcome = 'WIN';  break
            else:
                if bar['High'] >= sl: outcome = 'LOSS'; break
                if bar['Low']  <= tp: outcome = 'WIN';  break

        if outcome == 'OPEN': outcome = 'LOSS'
        pnl = risk_usd * rr if outcome == 'WIN' else -risk_usd
        if outcome == 'WIN': wins += 1
        else: losses += 1

        equity += pnl
        if equity > hwm:
            hwm = equity; floor = max(floor, hwm * 0.9)
        equity_curve.append(equity)
        results.append({'Dir': direction, 'Entry': entry, 'SL': sl, 'TP': tp,
                        'SL_Dist': round(sl_dist, 2), 'Outcome': outcome, 'Equity': equity})

        if equity <= floor:
            print(f"❌ CHÁY tại lệnh {wins+losses} | Equity: ${equity:,.0f}")
            break
        if equity >= STARTING_BAL * 1.08:
            print(f"✅ PASS QUỸ tại lệnh {wins+losses} | Equity: ${equity:,.0f}")
            break

        i += (bars_held or 1) + 1

    total = wins + losses
    if total == 0:
        print("[-] Không có lệnh nào."); return

    wr = wins / total
    ev = wr * rr - (1 - wr)
    print(f"\n{'='*60}")
    print(f"📊 XGBoost + SMC Signal Engine — M5 XAUUSD")
    print(f"{'='*60}")
    print(f"  Tổng lệnh : {total} | Win: {wins} | Loss: {losses}")
    print(f"  Winrate   : {wr*100:.1f}%")
    print(f"  EV/lệnh   : {ev:+.3f}R  {'✅ EV DƯƠNG!' if ev > 0 else '❌ EV âm'}")
    print(f"  SL        : Structural (Swing Low/High gần nhất, 2-5 pip)")
    print(f"  R:R       : 1:{rr} | Equity: ${equity_curve[-1]:,.0f}")
    print(f"{'='*60}")
    if results:
        print("\n📝 5 lệnh mẫu:")
        print(f"{'Dir':<7}{'Entry':>10}{'SL':>10}{'TP':>10}{'SL_Dist':>10}{'Kết Quả':>10}")
        for r in results[-5:]:
            print(f"  {r['Dir']:<7}{r['Entry']:>10.2f}{r['SL']:>10.2f}{r['TP']:>10.2f}{r['SL_Dist']:>10.2f}{r['Outcome']:>10}")

if __name__ == "__main__":
    print("[*] Load & Resample M5...")
    df = load_resample()
    print(f"[*] Base Features trên {len(df)} nến M5...")
    df = add_base_features(df)
    print("[*] SMC Feature Engineering...")
    df = calculate_all_smc(df, swing_lookback=5, dist_lookback=50)
    print("[*] Train XGBoost...")
    model, test_df = train_xgboost(df)
    print(f"[*] Backtest thực chiến (Conf={CONFIDENCE:.0%}, Structural SL, R:R 1:1.5)...")
    run_backtest(test_df)
