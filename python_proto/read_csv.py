import pandas as pd
import numpy as np
import os

def load_xauusd_data():
    """
    Đọc dữ liệu từ file CSV XAUUSD (2 năm 2024 và 2025)
    """
    print("[*] Đang đọc dữ liệu XAUUSD từ file CSV...")
    
    file_2024 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2024_all.csv"
    file_2025 = "d:/Cá nhân/Trading/Bot dự đoán cây nến tiếp theo/XAUUSD_2025_all.csv"
    
    if not os.path.exists(file_2024) or not os.path.exists(file_2025):
        print(f"[-] Không tìm thấy file data. Vui lòng kiểm tra lại đường dẫn.")
        return None

    # Đọc 2 file tổng (M1, M5 hay M15 thì tuỳ file export, pandas sẽ tự parse)
    # File MT5 xuất ra có thể dùng tab (\t) thay vì dấu phẩy
    try:
        df_2024 = pd.read_csv(file_2024, sep='\t' if '\t' in open(file_2024).readline() else ',')
        df_2025 = pd.read_csv(file_2025, sep='\t' if '\t' in open(file_2025).readline() else ',')
    except Exception as e:
         print(f"[-] Lỗi đọc file: {e}")
         return None
         
    # Nối 2 dataframe lại
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # Clean header (Loại bỏ các ký tự < >)
    df.columns = [c.strip().replace('<', '').replace('>', '') for c in df.columns]
    
    # Đổi tên cột cho khớp với code cũ
    rename_map = {
        'OPEN': 'Open',
        'HIGH': 'High',
        'LOW': 'Low',
        'CLOSE': 'Close',
        'TICKVOL': 'Volume',
        'VOL': 'Volume'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Có thể gộp Date và Time thành Datetime Index nếu cần
    if 'DATE' in df.columns and 'TIME' in df.columns:
        df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
        df.set_index('Datetime', inplace=True)
        # Sắp xếp lại theo thời gian
        df.sort_index(inplace=True)
        # Xoá cột thừa
        df.drop(columns=['DATE', 'TIME'], inplace=True, errors='ignore')
        
    print(f"[+] Đã tải xong {len(df)} nến XAUUSD.")
    return df

if __name__ == "__main__":
    df = load_xauusd_data()
    if df is not None:
         print("--- SAMPLE DATA XAUUSD ---")
         print(df.head(3))
