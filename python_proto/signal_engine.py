"""
signal_engine.py - Hệ thống Khuyến Nghị Lệnh XAUUSD Hoàn Chỉnh
Tác giả: hungpixi (Phạm Phú Nguyễn Hưng) | Comarai: https://comarai.com

Approach:
  Model 1 (KNN Classifier): Dự đoán DIRECTION (Long/Short)
  Model 2 (KNN Regressor):  Dự đoán VOLATILITY (ATR move size) → Tính SL/TP thực tế

Output mỗi nến:
  Signal:     LONG | SHORT | NO_TRADE
  Entry:      Giá mở cửa nến tiếp theo
  SL Price:   Entry - Volatility_Predicted × SL_Factor
  TP Price:   Entry + Volatility_Predicted × TP_Factor
  Confidence: % tự tin của model (dựa trên Prob từ KNN)

Tại sao predict Volatility?
  → Thay vì dùng ATR lịch sử cố định (lag), model học được thời điểm nào thị trường
    dao động mạnh/yếu để điều chỉnh SL/TP động theo từng lệnh.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

DATA_2024 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
DATA_2025 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"

# === CONFIG ===
TIMEFRAME    = 'M5'       # M1, M5, or M15
RESAMPLE_MAP = {'M1': None, 'M5': '5min', 'M15': '15min'}
CONFIDENCE_THRESHOLD = 0.62   # Chỉ ra signal khi model tự tin > 62%
SL_FACTOR    = 1.0            # SL = Predicted_ATR × SL_Factor
TP_FACTOR    = 1.5            # TP = Predicted_ATR × TP_Factor → R:R 1:1.5
K_NEIGHBORS  = 51
STARTING_BALANCE = 100_000
RISK_PER_TRADE   = 0.01       # 1% tài khoản / lệnh

def load_raw_m1():
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
    return df

def resample(df_m1, tf_str):
    """Gộp M1 sang timeframe bất kỳ"""
    return df_m1.resample(tf_str).agg(
        {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    ).dropna()

def feature_engineering(df):
    df = df.copy()
    # Trend
    df['SMA_50']   = df['Close'].rolling(50).mean()
    df['SMA_200']  = df['Close'].rolling(200).mean()
    df['Trend_Up'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)

    # Price Action (dễ port sang PineScript)
    df['Body']       = df['Close'] - df['Open']
    df['Body_Pct']   = (df['Body'] / df['Open']) * 100
    df['Range']      = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open','Close']].min(axis=1) - df['Low']

    # Volatility (ATR)
    df['ATR_14']   = df['Range'].rolling(14).mean()
    df['BodyATR']  = df['Body'].abs() / df['ATR_14']
    df['RangeMA']  = df['Range'] / df['ATR_14']  # Range tương đối để chuẩn hóa

    # Momentum
    delta = df['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14']   = 100 - (100 / (1 + g / l))
    df['ROC_5']    = df['Close'].pct_change(5) * 100

    # Volume
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # SESSION FILTER (UTC, XAUUSD liquidity)
    df['Hour'] = df.index.hour
    df['Is_London'] = ((df['Hour'] >= 7) & (df['Hour'] < 16)).astype(int)
    df['Is_NY']     = ((df['Hour'] >= 13) & (df['Hour'] < 21)).astype(int)
    df['Is_Active'] = np.where((df['Is_London'] == 1) | (df['Is_NY'] == 1), 1, 0)

    # === TARGET 1: DIRECTION - Nến tiếp theo Tăng (1) hay Giảm (0) ===
    df['Next_Body']   = df['Close'].shift(-1) - df['Open'].shift(-1)
    df['Target_Dir']  = np.where(df['Next_Body'] > 0, 1, 0)

    # === TARGET 2: VOLATILITY - Biên độ dao động (Range) của nến tiếp theo ===
    df['Target_Vol']  = (df['High'].shift(-1) - df['Low'].shift(-1))  # Range nến tiếp theo

    df.dropna(inplace=True)
    return df

FEATURES = ['Body_Pct', 'Upper_Wick', 'Lower_Wick', 'BodyATR', 'RangeMA',
            'RSI_14', 'ROC_5', 'Trend_Up', 'Vol_Ratio']

def train_models(df):
    df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ['Target_Dir', 'Target_Vol'])
    X  = df2[FEATURES].values
    y_dir = df2['Target_Dir'].values
    y_vol = df2['Target_Vol'].values

    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)
    split = int(len(X_s) * 0.8)

    # Model 1: Direction (Classifier)
    clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    clf.fit(X_s[:split], y_dir[:split])

    # Model 2: Volatility (Regressor)
    reg = KNeighborsRegressor(n_neighbors=K_NEIGHBORS)
    reg.fit(X_s[:split], y_vol[:split])

    return clf, reg, scaler, df2, split

def generate_signals(df2, clf, reg, scaler, split):
    """Tạo Signal với Entry/SL/TP từ model"""
    test_df = df2.iloc[split:].copy()
    X_test  = scaler.transform(test_df[FEATURES].values)

    prob_up   = clf.predict_proba(X_test)[:, 1]
    pred_vol  = reg.predict(X_test)

    test_df['Prob_Up']  = prob_up
    test_df['Pred_Vol'] = pred_vol

    # Tính SL/TP dùng Predicted Volatility (Dynamic)
    signals = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        entry = row['Open']
        pred_atr = max(row['Pred_Vol'], row['ATR_14'] * 0.5)  # Đảm bảo SL ít nhất 0.5 × ATR

        if row['Prob_Up'] >= CONFIDENCE_THRESHOLD and row['Is_Active'] == 1 and row['Trend_Up'] == 1:
            sl = entry - pred_atr * SL_FACTOR
            tp = entry + pred_atr * TP_FACTOR
            signals.append({
                'Time'      : idx,
                'Signal'    : 'LONG',
                'Entry'     : round(entry, 2),
                'SL'        : round(sl, 2),
                'TP'        : round(tp, 2),
                'Confidence': round(row['Prob_Up'] * 100, 1),
                'Pred_ATR'  : round(pred_atr, 2),
                'Actual_Dir': int(row['Target_Dir']),
                'Actual_Vol': round(row['Target_Vol'], 2),
            })
        elif row['Prob_Up'] <= (1 - CONFIDENCE_THRESHOLD) and row['Is_Active'] == 1 and row['Trend_Up'] == -1:
            sl = entry + pred_atr * SL_FACTOR
            tp = entry - pred_atr * TP_FACTOR
            signals.append({
                'Time'      : idx,
                'Signal'    : 'SHORT',
                'Entry'     : round(entry, 2),
                'SL'        : round(sl, 2),
                'TP'        : round(tp, 2),
                'Confidence': round((1 - row['Prob_Up']) * 100, 1),
                'Pred_ATR'  : round(pred_atr, 2),
                'Actual_Dir': int(row['Target_Dir']),
                'Actual_Vol': round(row['Target_Vol'], 2),
            })

    return pd.DataFrame(signals)

def backtest_signals(signals_df, df_full):
    """Backtest: Với mỗi signal, tìm xem SL hay TP được chạm trước trong các nến tiếp theo"""
    if signals_df.empty:
        print("[-] Không có signal nào!")
        return

    equity   = STARTING_BALANCE
    hwm      = STARTING_BALANCE
    floor    = STARTING_BALANCE * 0.9  # 10% trailing DD
    risk_usd = STARTING_BALANCE * RISK_PER_TRADE

    wins = losses = 0
    equity_curve = []
    results = []

    df_idx = df_full.index.tolist()

    for _, sig in signals_df.iterrows():
        try:
            sig_pos = df_idx.index(sig['Time'])
        except ValueError:
            continue

        # Duyệt tối đa 20 nến tiếp theo tìm SL/TP
        outcome = 'OPEN'
        for j in range(sig_pos + 1, min(sig_pos + 21, len(df_idx))):
            bar = df_full.iloc[j]
            if sig['Signal'] == 'LONG':
                if bar['Low']  <= sig['SL']: outcome = 'LOSS'; break
                if bar['High'] >= sig['TP']: outcome = 'WIN';  break
            else:
                if bar['High'] >= sig['SL']: outcome = 'LOSS'; break
                if bar['Low']  <= sig['TP']: outcome = 'WIN';  break

        if outcome == 'OPEN':
            outcome = 'LOSS'  # Chưa hit sau 20 nến → tính loss (conservative)

        pnl = risk_usd * TP_FACTOR if outcome == 'WIN' else -risk_usd
        if outcome == 'WIN': wins += 1
        else: losses += 1

        equity += pnl
        if equity > hwm:
            hwm   = equity
            floor = max(floor, hwm * 0.9)

        equity_curve.append(equity)
        results.append({'Time': sig['Time'], 'Signal': sig['Signal'],
                        'Entry': sig['Entry'], 'SL': sig['SL'], 'TP': sig['TP'],
                        'Confidence': sig['Confidence'], 'Outcome': outcome, 'Equity': equity})

        if equity <= floor:
            print(f"❌ CHÁY MAX DD tại lệnh [{wins+losses}] | Equity: ${equity:,.0f}")
            break
        if equity >= STARTING_BALANCE * 1.08:
            print(f"✅ PASS QUỸ tại lệnh [{wins+losses}] | Equity: ${equity:,.0f}")
            break

    total = wins + losses
    wr = wins / total if total > 0 else 0
    ev = wr * TP_FACTOR - (1 - wr)

    print(f"\n{'='*60}")
    print(f"📊 SIGNAL ENGINE BACKTEST — {TIMEFRAME} XAUUSD")
    print(f"{'='*60}")
    print(f"  Tổng Signal      : {total} | Win: {wins} | Loss: {losses}")
    print(f"  Winrate Thực     : {wr*100:.1f}%")
    print(f"  EV/lệnh          : {ev:+.3f}R  {'✅ Profitable' if ev > 0 else '❌ Thua dài hạn'}")
    print(f"  Equity Cuối      : ${equity_curve[-1] if equity_curve else STARTING_BALANCE:,.0f}")
    print(f"  SL Factor        : {SL_FACTOR}× ATR_pred | TP Factor: {TP_FACTOR}× ATR_pred")
    print(f"  Confidence Filter: > {CONFIDENCE_THRESHOLD*100:.0f}% | Session: London + NY")
    print(f"{'='*60}")

    # In 5 signal mẫu gần nhất
    print(f"\n📝 5 SIGNAL MẪU GẦN NHẤT:")
    print(f"{'Time':28} {'Dir':8} {'Entry':10} {'SL':10} {'TP':10} {'Conf':8} {'Result':8}")
    for r in results[-5:]:
        print(f"  {str(r['Time']):26} {r['Signal']:8} {r['Entry']:10.2f} {r['SL']:10.2f} {r['TP']:10.2f} {r['Confidence']:6.1f}%  {r['Outcome']:8}")

if __name__ == "__main__":
    print(f"[*] Load M1 data (710k nến)...")
    df_m1 = load_raw_m1()

    if RESAMPLE_MAP[TIMEFRAME]:
        print(f"[*] Resample M1 → {TIMEFRAME}...")
        df = resample(df_m1, RESAMPLE_MAP[TIMEFRAME])
    else:
        df = df_m1.copy()

    print(f"[*] Feature Engineering ({TIMEFRAME}: {len(df)} nến)...")
    df_feat = feature_engineering(df)

    print(f"[*] Train KNN Direction + KNN Volatility Regressor (K={K_NEIGHBORS})...")
    clf, reg, scaler, df2, split = train_models(df_feat)

    print(f"[*] Generate Signals (Threshold={CONFIDENCE_THRESHOLD*100:.0f}%, {TIMEFRAME}, London+NY)...")
    signals = generate_signals(df2, clf, reg, scaler, split)

    if signals.empty:
        print("[-] Không có signal nào thỏa mãn. Hãy giảm CONFIDENCE_THRESHOLD.")
    else:
        print(f"[+] Tổng số signal: {len(signals)} | Long: {(signals['Signal']=='LONG').sum()} | Short: {(signals['Signal']=='SHORT').sum()}")
        per_month = len(signals) / (len(df2.iloc[split:]) / (4 * 24 * 22) if TIMEFRAME == 'M15' else
                                     len(df2.iloc[split:]) / (60 * 24 * 22) if TIMEFRAME == 'M1' else
                                     len(df2.iloc[split:]) / (12 * 24 * 22))
        print(f"[+] Trung bình: ~{per_month:.0f} signal/tháng")
        backtest_signals(signals, df.iloc[split + df_feat.index.get_loc(df2.index[split]) - len(df2.index[:split]):])
