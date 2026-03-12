import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def get_xauusd_data():
    """ Đọc từ file CSV XAUUSD M1 cục bộ """
    print("[*] Load XAUUSD MT5 Data...")
    file_2024 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
    file_2025 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"
    
    # Định dạng chuẩn file MT5: Ngày Thời_gian Open High Low Close Volume
    cols = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # File thường không có header, và cách nhau bằng dấu phẩy
    df_24 = pd.read_csv(file_2024, sep=',', names=cols)
    df_25 = pd.read_csv(file_2025, sep=',', names=cols)
    
    df = pd.concat([df_24, df_25], ignore_index=True)
    
    # Chuyển kiểu dữ liệu
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].astype(float)
        
    df.dropna(inplace=True)
    print(f"[+] Loaded: {len(df)} candles.")
    return df

def feature_engineering(df):
    """ Feature Engineering & Lọc Xu Hướng """
    print("[*] Feature Engineering M1 Data...")
    
    # Do M1 nhiễu hơn H1 rất nhiều, ta cần bộ lọc Trend dài hạn (VD: SMA 100 và SMA 300)
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_300'] = df['Close'].rolling(window=300).mean()
    df['Trend_Up'] = np.where(df['SMA_100'] > df['SMA_300'], 1, -1)
    
    df['Body'] = df['Close'] - df['Open']
    df['Body_Pct'] = (df['Body'] / df['Open']) * 100
    
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Thêm Volatility / ATR nhẹ
    df['ATR_20'] = (df['High'] - df['Low']).rolling(20).mean()
    df['Body_to_ATR'] = df['Body'].abs() / df['ATR_20']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Target (Shift -1: Lấy nến tiếp theo)
    df['Next_Candle_Return'] = df['Close'].shift(-1) - df['Open'].shift(-1)
    df['Target'] = np.where(df['Next_Candle_Return'] > 0, 1, 0)
    
    df.dropna(inplace=True)
    return df

def train_and_backtest_knn(df):
    print("\n[*] Đang Train KNN Model (M1 Data - 700k+ candles)...")
    
    # Thu hẹp lại cỡ mẫu (Dùng 200,000 nến gần nhất để code chạy nhanh, tương đương ~400 ngày M1)
    df = df.tail(200000).copy()
    
    features = ['Body_Pct', 'Upper_Shadow', 'Lower_Shadow', 'Body_to_ATR', 'RSI_14', 'Trend_Up']
    
    # Xóa vô hạn
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['Target'])
    X = df[features].values
    y = df['Target'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train 80 / Test 20
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    k_neighbors = 144
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    
    # Dự đoán Test
    y_prob = model.predict_proba(X_test)
    test_df = df.iloc[split_idx:].copy()
    test_df['Prob_Up'] = y_prob[:, 1]
    
    # --- LỌC NGƯỠNG TỰ TIN VÀ TREND ---
    threshold = 0.65
    test_df['Signal'] = np.where(test_df['Prob_Up'] >= threshold, 1, 
                          np.where(test_df['Prob_Up'] <= (1 - threshold), 0, np.nan))
    
    # Đánh thuận xu hướng
    test_df['Final_Signal'] = np.where(
        (test_df['Signal'] == 1) & (test_df['Trend_Up'] == 1), 1,
        np.where((test_df['Signal'] == 0) & (test_df['Trend_Up'] == -1), 0, np.nan)
    )
    
    total_signals = test_df['Final_Signal'].notna().sum()
    if total_signals == 0:
         print("[-] Quá ít tín hiệu. Vui lòng giảm Threshold.")
         return
         
    # Check lệnh Thắng (Target = Final_Signal)
    test_df['Is_Win'] = np.where(test_df['Final_Signal'] == test_df['Target'], 1, 0)
    
    trades_taken = test_df.dropna(subset=['Final_Signal'])
    actual_wins = trades_taken['Is_Win'].sum()
    actual_winrate = (actual_wins / total_signals) * 100
    
    print("\n--- KẾT QUẢ BACKTEST XAUUSD M1 MỚI NHẤT (Ngưỡng 65%, Cùng Trend) ---")
    print(f"Tổng số nến Test: {len(test_df)}")
    print(f"Số Lệnh được đánh: {total_signals} lệnh")
    print(f"Số win: {actual_wins} / Số loss: {total_signals - actual_wins}")
    print(f"WINRATE RÒNG: {actual_winrate:.2f}%")

if __name__ == "__main__":
    df = get_xauusd_data()
    df_prepared = feature_engineering(df)
    train_and_backtest_knn(df_prepared)
