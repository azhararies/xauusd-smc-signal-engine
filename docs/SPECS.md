# SPECS: Hybrid ML Next-Candle Predictor

## 1. Executive Summary
Hệ thống dự đoán xác suất Tăng/Giảm của cây nến tiếp theo (Next-Candle) thông qua thuật toán phân loại (ví dụ: K-Nearest Neighbors - KNN). Phát triển qua 2 giai đoạn: Prototype trên Python (để test và tìm Winrate 70%) và Production/Native trên PineScript.

## 2. Giai đoạn 1: Python Engine (Prototype & Backtest)
- **Input Data:** Lịch sử giá OHLCV (Ví dụ: dùng MetaTrader5 thư viện Python hoặc thả file CSV).
- **Tính toán Features (Pandas/TA-Lib):**
  - RSI (Relative Strength Index).
  - Độ lệch chuẩn (Volatility).
  - Khối lượng giao dịch (Volume Change).
- **Thuật toán (Scikit-learn):** `KNeighborsClassifier`.
  - Train model dựa trên N cây nến gần nhất (Sliding Window hoặc Fixed).
  - `y` (Target) = 1 (Tăng) hoặc 0 (Giảm).
- **Kết quả trả về:** Script Python phải in ra terminal được báo cáo Winrate rõ ràng.

## 3. Giai đoạn 2: PineScript Translation
- Từ công thức chuẩn hóa và tập Features lấy từ Python, dịch logic 1-1 sang Pine v5.
- Ở mỗi bar, tính toán mảng đặc trưng tương ứng, lưu vào Array (Lookback 2000 bars) và đếm khoảng cách Minkowski/Euclidean thủ công trong Pine.

## 4. UI/UX trên Chart (Sau khi chuyển Pine)
- Bảng Table góc màn hình hiển thị tỉ lệ %.

## 5. GitHub Portfolio & Branding
- Repo: `ML-NextCandle-Python2Pine`.
- Cấu trúc thư mục phải sạch: 
  - `/python_proto/` (cho code file ipynb, py).
  - `/pinescript/` (cho code v5).
- Tích hợp CTA Comarai Services (Demo, Zalo, Email).
