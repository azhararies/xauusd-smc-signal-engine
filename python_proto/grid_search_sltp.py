"""
grid_search_sltp.py - Tự động tìm SL/TP Factor tối ưu cho EV dương
Chạy 1 lần duy nhất để tìm ra cấu hình tốt nhất, không cần đoán mò.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

DATA_2024 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
DATA_2025 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"

TIMEFRAME         = 'M5'      # Có thể đổi M15 hoặc M1
CONFIDENCE        = 0.62
K                 = 51
MIN_SL_PRICE_DIST = 3.0       # Tối thiểu 30 pip (3 price points XAUUSD)
MAX_BARS_HOLD     = 30        # Tối đa 30 nến M5 = 2.5h

FEATURES = ['Body_Pct', 'Upper_Wick', 'Lower_Wick', 'BodyATR', 'RangeMA',
            'RSI_14', 'ROC_5', 'Trend_Up', 'Vol_Ratio']

def load_and_prepare():
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
    # Resample → M5
    df = df.resample('5min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    df['SMA_50']     = df['Close'].rolling(50).mean()
    df['SMA_200']    = df['Close'].rolling(200).mean()
    df['Trend_Up']   = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Body']       = df['Close'] - df['Open']
    df['Body_Pct']   = (df['Body'] / df['Open']) * 100
    df['Range']      = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['ATR_14']     = df['Range'].rolling(14).mean()
    df['BodyATR']    = df['Body'].abs() / df['ATR_14']
    df['RangeMA']    = df['Range'] / df['ATR_14']
    delta = df['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14']     = 100 - (100 / (1 + g / l))
    df['ROC_5']      = df['Close'].pct_change(5) * 100
    df['Vol_Ratio']  = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Hour']       = df.index.hour
    df['Is_Active']  = np.where(((df['Hour'] >= 7) & (df['Hour'] < 21)), 1, 0)
    df['Target_Vol'] = (df['High'].shift(-1) - df['Low'].shift(-1))
    df['Target_Dir'] = np.where((df['Close'].shift(-1) - df['Open'].shift(-1)) > 0, 1, 0)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)   # Giữ Datetime như cột thường để duyệt nhanh
    return df

def train(df):
    df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ['Target_Dir', 'Target_Vol'])
    df2.reset_index(drop=True, inplace=True)
    X = df2[FEATURES].values
    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)
    split = int(len(X_s) * 0.8)
    clf = KNeighborsClassifier(n_neighbors=K).fit(X_s[:split], df2['Target_Dir'].values[:split])
    reg = KNeighborsRegressor(n_neighbors=K).fit(X_s[:split], df2['Target_Vol'].values[:split])
    prob = clf.predict_proba(X_s[split:])[:, 1]
    vol  = reg.predict(X_s[split:])
    test = df2.iloc[split:].copy().reset_index(drop=True)
    test['Prob_Up'] = prob
    test['Pred_Vol'] = vol
    print(f"[+] Test set: {len(test)} nến M5 | ATR trung bình: {test['ATR_14'].mean():.2f}")
    return test

def run_backtest_with_factors(test_df, sl_f, rr):
    """Chạy backtest nhanh với 1 bộ SL_Factor + RR"""
    wins = losses = 0
    i = 0
    while i < len(test_df) - MAX_BARS_HOLD:
        row = test_df.iloc[i]
        if row['Is_Active'] == 0:
            i += 1; continue

        entry = row['Open']
        pred_atr = max(row['Pred_Vol'], row['ATR_14'])
        sl_dist  = pred_atr * sl_f
        
        # Bỏ qua nếu SL quá nhỏ (dưới 30 pip)
        if sl_dist < MIN_SL_PRICE_DIST:
            i += 1; continue

        if row['Prob_Up'] >= CONFIDENCE and row['Trend_Up'] == 1:
            direction = 'L'
            sl = entry - sl_dist
            tp = entry + sl_dist * rr
        elif row['Prob_Up'] <= 1 - CONFIDENCE and row['Trend_Up'] == -1:
            direction = 'S'
            sl = entry + sl_dist
            tp = entry - sl_dist * rr
        else:
            i += 1; continue

        outcome = 'OPEN'
        bars_held = 0
        for j in range(i + 1, min(i + 1 + MAX_BARS_HOLD, len(test_df))):
            bar = test_df.iloc[j]
            bars_held = j - i
            if direction == 'L':
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
    if total < 10:
        return None   # Không đủ lệnh để đánh giá
    wr = wins / total
    ev = wr * rr - (1 - wr)
    return {'sl_f': sl_f, 'rr': rr, 'wins': wins, 'losses': losses, 'total': total,
            'winrate': wr, 'ev': ev}

if __name__ == "__main__":
    print("[*] Load & Prepare M5 data...")
    df = load_and_prepare()
    print("[*] Train KNN Models...")
    test_df = train(df)

    # Grid Search SL Factor × R:R
    sl_factors = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    rr_values  = [1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"\n{'='*65}")
    print(f" GRID SEARCH: SL_Factor × R:R (SL tối thiểu ≥ {MIN_SL_PRICE_DIST} price pts)")
    print(f"{'='*65}")
    print(f"{'SL_F':>6} | {'R:R':>5} | {'Lệnh':>6} | {'WR%':>7} | {'EV/trade':>10} | {'Đánh giá':>12}")
    print('-' * 65)

    results = []
    for sl_f in sl_factors:
        for rr in rr_values:
            r = run_backtest_with_factors(test_df, sl_f, rr)
            if r is None:
                continue
            ev_str = f"{r['ev']:+.3f}R"
            rating = "✅ EV DƯƠNG!" if r['ev'] > 0.05 else ("⚠️ Hòa vốn" if r['ev'] > -0.05 else "❌ EV âm")
            print(f"  {sl_f:>4.1f}× | {rr:>4.1f} | {r['total']:>6} | {r['winrate']*100:>6.1f}% | {ev_str:>10} | {rating}")
            results.append(r)

    # Tìm cấu hình tốt nhất theo EV
    positive_ev = [r for r in results if r['ev'] > 0]
    if positive_ev:
        best = max(positive_ev, key=lambda x: x['ev'])
        print(f"\n{'='*65}")
        print(f"🏆 TỐT NHẤT: SL = {best['sl_f']}× ATR | R:R = 1:{best['rr']}")
        print(f"   Winrate: {best['winrate']*100:.1f}% | EV: {best['ev']:+.3f}R/lệnh")
        print(f"   Tổng lệnh: {best['total']} (Win:{best['wins']}, Loss:{best['losses']})")
        print(f"   Với ATR M5 trung bình ~2.5 pip → SL ~{best['sl_f']*2.5:.1f} price points")
        print(f"{'='*65}")
    else:
        print(f"\n❌ Chưa tìm được EV dương ở bất kỳ combo nào!")
        best_attempt = max(results, key=lambda x: x['ev'])
        print(f"   Gần nhất: SL={best_attempt['sl_f']}× | R:R=1:{best_attempt['rr']} | EV={best_attempt['ev']:+.3f}R")
