# Phase 02: Feature Engineering & Math
Status: ⬜ Pending
Dependencies: Phase 01

## Objective
Xử lý dữ liệu đầu vào và triển khai cốt lõi toán học của K-Nearest Neighbors.

## Implementation Steps
1. [ ] Code hàm tính toán các Features (Volatility, RSI, Volume ROC).
2. [ ] Hàm chuẩn hóa dữ liệu (Min-Max Scaling) để khoảng cách tính toán đồng đều.
3. [ ] Viết vòng lặp tính khoảng cách (Euclidean/Lorentzian distance) giữa nến hiện tại và nến trong quá khứ.
4. [ ] Thuật toán sort và chọn ra K nến gần nhất, từ đó bỏ phiếu (Majority Voting) nến tiếp theo Tăng hay Giảm.

## Files to Create/Modify
- `ML_NextCandle_Predictor.pine` (Cập nhật lõi thuật toán).
