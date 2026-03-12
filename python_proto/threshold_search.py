"""Grid Search tìm Threshold tối ưu cho Pipeline KNN XAUUSD M1"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

DATA_2024 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
DATA_2025 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"

cols = ['Date','Time','Open','High','Low','Close','Volume']
df24 = pd.read_csv(DATA_2024, sep=',', names=cols)
df25 = pd.read_csv(DATA_2025, sep=',', names=cols)
df = pd.concat([df24, df25], ignore_index=True)
for c in ['Open','High','Low','Close']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df.dropna(inplace=True)

df['SMA_100'] = df['Close'].rolling(100).mean()
df['SMA_300'] = df['Close'].rolling(300).mean()
df['Trend_Up'] = np.where(df['SMA_100'] > df['SMA_300'], 1, -1)
df['Body'] = df['Close'] - df['Open']
df['Body_Pct'] = (df['Body'] / df['Open']) * 100
df['Upper'] = df['High'] - df[['Open','Close']].max(axis=1)
df['Lower'] = df[['Open','Close']].min(axis=1) - df['Low']
df['ATR_20'] = (df['High'] - df['Low']).rolling(20).mean()
df['BodyATR'] = df['Body'].abs() / df['ATR_20']
delta = df['Close'].diff()
g = (delta.where(delta > 0, 0)).rolling(14).mean()
l = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI_14'] = 100 - (100 / (1 + g / l))
df['Next_Ret'] = df['Close'].shift(-1) - df['Open'].shift(-1)
df['Target'] = np.where(df['Next_Ret'] > 0, 1, 0)
df.dropna(inplace=True)

features = ['Body_Pct', 'Upper', 'Lower', 'BodyATR', 'RSI_14', 'Trend_Up']
df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['Target'])
X = df2[features].values
y = df2['Target'].values
scaler = MinMaxScaler()
X_s = scaler.fit_transform(X)
split = int(len(X_s) * 0.8)
model = KNeighborsClassifier(n_neighbors=144)
model.fit(X_s[:split], y[:split])
prob = model.predict_proba(X_s[split:])[:, 1]
test_df = df2.iloc[split:].copy()
test_df['Prob_Up'] = prob

print(f'--- GRID SEARCH THRESHOLD ---')
print(f'Threshold | Số Lệnh | Winrate | ~Lệnh/Tháng')
print('-'*55)
for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]:
    buy = (test_df['Prob_Up'] >= thr) & (test_df['Trend_Up'] == 1)
    sell = (test_df['Prob_Up'] <= 1 - thr) & (test_df['Trend_Up'] == -1)
    signals = test_df[buy | sell].copy()
    if len(signals) == 0:
        print(f'  {thr:.0%}   |   0     |  N/A    |  N/A')
        continue
    signals['Signal'] = np.where(signals['Prob_Up'] >= thr, 1, 0)
    wr = (signals['Signal'] == signals['Target']).mean()
    per_month = len(signals) / 6  # Test set ~6 tháng
    print(f'  {thr:.0%}   |  {len(signals):4d}   |  {wr*100:.1f}%  |  ~{per_month:.0f} lenh/thang')
