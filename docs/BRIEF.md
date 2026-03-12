# 💡 BRIEF: Machine Learning Predictor (Next-Candle)

**Tác giả:** Phạm Phú Nguyễn Hưng (hungpixi)  
**Ngày tạo:** 2026-03-12  
**Mục đích:** Xây dựng hệ thống dự đoán nến tiếp theo bằng Machine Learning. Hỗ trợ trade quỹ và làm Portfolio trên GitHub quảng bá Comarai.com.

## 1. VẤN ĐỀ CẦN GIẢI QUYẾT
- Các indicator truyền thống thường bị trễ (lagging).
- Trader cần một góc nhìn thống kê: "Trong quá khứ, dữ liệu tương tự dẫn đến nến Tăng hay Giảm?".
- *Quyết định mới:* Thay vì code PineScript thuần túy ngay từ đầu (khó debug, hạn chế data), hệ thống sẽ được xây và test bằng **Python** trước để đánh giá mức độ khả thi và lấy winrate 70%, sau đó mới port sang PineScript.

## 2. GIẢI PHÁP ĐỀ XUẤT (HYBRID APPROACH)
- Xây dựng mô hình KNN bằng Python (Sử dụng Thư viện Scikit-learn/Pandas).
- Lấy data chuẩn từ thư viện (như `yfinance` hoặc `MetaTrader5` API).
- Train và Backtest ngay trên Python.
- Sau khi chốt được công thức/Features tối ưu -> Viết lại 100% logic đó sang PineScript v5 (như một Indicator độc lập) hoặc giữ Python chạy qua Webhook.

## 3. NGHIÊN CỨU THỊ TRƯỜNG & KHÁC BIỆT
- **Điểm khác biệt (Portfolio của hungpixi):** 
  - Thể hiện được tư duy xây dựng Agent Xuyên Nền Tảng (Python -> TradingView).
  - Bản code tối ưu performance, dễ đọc, không rườm rà. Bổ sung các bộ lọc xu hướng gốc. Gắn nhận diện thương hiệu Comarai.

## 4. TÍNH NĂNG CỐT LÕI (MVP)
1.  **Python Engine:**
    - Bot lấy dữ liệu OHLCV.
    - Feature Engineering cơ bản (RSI, ROC, Moving Averages).
    - KNN Model trả ra kết quả Backtest (Winrate, R:R).
2.  **PineScript Indicator:**
    - Cổng kết nối (nếu xài webhook) hoặc Code Native Pine (nếu Winrate đã thỏa mãn và thuật toán đơn giản).
