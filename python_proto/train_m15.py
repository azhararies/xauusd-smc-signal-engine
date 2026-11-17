"""
train_m15.py - Train KNN Model trên M15 data (Gộp từ M1 710k nến)
Lý do dùng M15 thay M1:
- M1 có quá nhiều noise, KNN khó tìm pattern ổn định
- M15 mỗi nến tóm tắt 15 phút thị trường -> ít noise hơn, feature ý nghĩa hơn
- SL/TP theo Zigzag M15 xa hơn, ít bị quét bởi noise spread
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

M15_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "XAUUSD_M15.csv")

def load_m15():
    df = pd.read_csv(M15_CSV, index_col='Datetime', parse_dates=True)
    return df

def feature_engineering(df):
    df = df.copy()
    # Trend filter (dùng SMA ngắn hơn vì M15 là 7*100=700 nến M1)
    df['SMA_50']  = df['Close'].rolling(50).mean()   # ~12.5h
    df['SMA_200'] = df['Close'].rolling(200).mean()  # ~50h
    df['Trend_Up'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    
    # Price Action Features
    df['Body']      = df['Close'] - df['Open']
    df['Body_Pct']  = (df['Body'] / df['Open']) * 100
    df['Upper']     = df['High'] - df[['Open','Close']].max(axis=1)
    df['Lower']     = df[['Open','Close']].min(axis=1) - df['Low']
    df['Range']     = df['High'] - df['Low']
    df['ATR_14']    = df['Range'].rolling(14).mean()
    df['BodyATR']   = df['Body'].abs() / df['ATR_14']
    
    # Momentum
    delta         = df['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14']  = 100 - (100 / (1 + g / l))
    df['ROC_5']   = df['Close'].pct_change(5) * 100   # Rate of Change 5 nến = 1.25h
    
    # Volume (Tick Volume từ MT5)
    df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA_20']  # > 1 = volume cao bất thường

    # Target: Nến M15 tiếp theo tăng (1) hay giảm (0)?
    df['Next_Ret'] = df['Close'].shift(-1) - df['Open'].shift(-1)
    df['Target']   = np.where(df['Next_Ret'] > 0, 1, 0)
    
    df = df.dropna()
    return df

def find_swing_low(df_window, lookback=10):
    """Tìm Swing Low gần nhất trong cửa sổ quan sát trước đó"""
    if len(df_window) < lookback:
        return df_window['Low'].min()
    for i in range(len(df_window) - 1, lookback // 2, -1):
        center = i
        left   = max(0, center - lookback // 2)
        right  = min(len(df_window), center + lookback // 2)
        if df_window.iloc[center]['Low'] == df_window.iloc[left:right]['Low'].min():
            return df_window.iloc[center]['Low']
    return df_window['Low'].min()

def find_swing_high(df_window, lookback=10):
    """Tìm Swing High gần nhất trong cửa sổ quan sát trước đó"""
    if len(df_window) < lookback:
        return df_window['High'].max()
    for i in range(len(df_window) - 1, lookback // 2, -1):
        center = i
        left   = max(0, center - lookback // 2)
        right  = min(len(df_window), center + lookback // 2)
        if df_window.iloc[center]['High'] == df_window.iloc[left:right]['High'].max():
            return df_window.iloc[center]['High']
    return df_window['High'].max()

def run():
    print("[*] Load M15 data (resample từ M1 710k nến)...")
    df = load_m15()
    print(f"[+] M15: {len(df)} nến | {df.index[0]} → {df.index[-1]}")

    print("[*] Feature Engineering M15...")
    df = feature_engineering(df)
    print(f"[+] Sau FE: {len(df)} nến hợp lệ")

    features = ['Body_Pct', 'Upper', 'Lower', 'BodyATR', 'RSI_14', 'ROC_5', 'Trend_Up', 'Vol_Ratio']
    df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['Target'])

    X = df2[features].values
    y = df2['Target'].values
    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)
    split = int(len(X_s) * 0.8)

    print(f"[*] Train KNN M15 (K=51) trên {split} nến...")
    model = KNeighborsClassifier(n_neighbors=51)
    model.fit(X_s[:split], y[:split])

    prob = model.predict_proba(X_s[split:])[:, 1]
    test_df = df2.iloc[split:].copy()
    test_df['Prob_Up'] = prob

    # === GRID SEARCH THRESHOLD trên M15 ===
    print("\n--- GRID SEARCH THRESHOLD (M15) ---")
    print(f"{'Thr':>6} | {'Lệnh':>6} | {'Winrate':>8} | {'Lệnh/Tháng':>12}")
    print('-' * 45)
    # Số tháng thực tế của test set, tính từ ngày đầu → ngày cuối
    N_MONTHS = (test_df.index[-1] - test_df.index[0]).total_seconds() / (86400 * 30.44)
    print(f"Test set: {test_df.index[0]} → {test_df.index[-1]} ({N_MONTHS:.2f} tháng)")
    for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        buy  = (test_df['Prob_Up'] >= thr) & (test_df['Trend_Up'] == 1)
        sell = (test_df['Prob_Up'] <= 1 - thr) & (test_df['Trend_Up'] == -1)
        sigs = test_df[buy | sell].copy()
        if len(sigs) == 0:
            print(f"  {thr:.0%}  |     0  |     N/A  |          N/A")
            continue
        sigs['Signal'] = np.where(sigs['Prob_Up'] >= thr, 1, 0)
        wr = (sigs['Signal'] == sigs['Target']).mean()
        pm = len(sigs) / N_MONTHS
        print(f"  {thr:.0%}  | {len(sigs):6d} | {wr*100:7.1f}%  | ~{pm:10.0f} lenh/thang")

if __name__ == "__main__":
    run()
