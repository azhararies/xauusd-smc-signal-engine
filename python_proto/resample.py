"""
resample.py - Gộp dữ liệu M1 XAUUSD sang M15 và H1
Dùng pandas resample - cực nhanh, không cần vòng lặp.

Quy tắc OHLCV khi gộp nến:
- Open  = Mở cửa của cây nến ĐẦU trong nhóm (first)
- High  = Cao nhất trong toàn bộ nhóm (max)
- Low   = Thấp nhất trong toàn bộ nhóm (min)
- Close = Đóng cửa của cây nến CUỐI trong nhóm (last)
- Volume = Tổng khối lượng (sum)
"""
import pandas as pd
import numpy as np

DATA_2024 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
DATA_2025 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"

def load_m1():
    cols = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df24 = pd.read_csv(DATA_2024, sep=',', names=cols)
    df25 = pd.read_csv(DATA_2025, sep=',', names=cols)
    df = pd.concat([df24, df25], ignore_index=True)
    
    # Chuyển Date + Time thành DateTime index - đây là bước quan trọng nhất
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df

def resample_ohlcv(df_m1, timeframe='15min'):
    """
    Gộp M1 sang bất kỳ timeframe nào.
    timeframe: '15min' cho M15, '1h' cho H1, '4h' cho H4, 'D' cho Daily
    """
    agg_dict = {
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum'
    }
    df_resampled = df_m1.resample(timeframe).agg(agg_dict)
    # Bỏ các nến trống (không có giao dịch - cuối tuần, nghỉ lễ)
    df_resampled.dropna(inplace=True)
    return df_resampled

if __name__ == "__main__":
    print("[*] Load M1 data...")
    df_m1 = load_m1()
    print(f"[+] M1: {len(df_m1):,} nến | {df_m1.index[0]} → {df_m1.index[-1]}")
    
    print("\n[*] Resample → M15...")
    df_m15 = resample_ohlcv(df_m1, '15min')
    print(f"[+] M15: {len(df_m15):,} nến")
    print(df_m15.tail(3))
    
    print("\n[*] Resample → H1...")
    df_h1 = resample_ohlcv(df_m1, '1h')
    print(f"[+] H1: {len(df_h1):,} nến")
    print(df_h1.tail(3))
    
    # Lưu ra CSV để dùng cho training sau
    df_m15.to_csv("d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_M15.csv")
    df_h1.to_csv("d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_H1.csv")
    print("\n[+] Đã lưu XAUUSD_M15.csv và XAUUSD_H1.csv")
