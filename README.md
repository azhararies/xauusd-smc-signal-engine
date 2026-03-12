# 🏆 XAUUSD SMC Signal Engine
> *Từ KNN thuần → XGBoost + Smart Money Concepts → EV dương thực chiến*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?logo=xgboost)](https://xgboost.readthedocs.io)
[![MT5](https://img.shields.io/badge/MetaTrader-5-blue)](https://www.metatrader5.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Comarai](https://img.shields.io/badge/By-Comarai-purple)](https://comarai.com)

---

## 📊 Kết Quả Thực Chiến (Walk-Forward 18 Tháng)

| Chỉ Số | Kết Quả |
|--------|---------|
| ✅ **EV/lệnh** | **+0.183R** (Dương — Profitable dài hạn) |
| 📈 **Winrate** | 47.3% (1,412 lệnh) |
| 📅 **Tháng EV+** | **14/18 tháng (78%)** |
| 🛡️ **Pass Quỹ** | **82%** (Monte Carlo 10,000 sims, Risk 1%) |
| 💰 **DD Trung Bình** | 5.5% (ngưỡng quỹ 10%) |
| 📉 **R:R** | 1:1.5 với Structural SL |
| ⏱️ **Tín hiệu** | ~78 lệnh/tháng trên M5 |

---

## 🧠 Tư Duy Thiết Kế — Điểm Khác Biệt

### ❌ Vấn Đề Với Các Approach Cũ

Hầu hết bot dự đoán nến dùng:
```
KNN(RSI, Body, ATR) → Predict UP/DOWN → Set SL/TP cứng
```
**Thực tế:** Grid Search 30 combo SL/TP trên M5 → **100% EV âm**. Tại sao?

- **Direction accuracy ≠ Trading winrate**: Đoán đúng 58% hướng nến nhưng SL/TP thực tế chỉ hit 33-44%
- **SL quá gần**: ATR × N trên M1/M5 = noise stop-out liên tục
- **KNN curse of dimensionality**: Hiệu suất giảm khi có nhiều features

### ✅ Giải Pháp: XGBoost + Smart Money Concepts

```
SMC Features (Order Block + FVG + Swing Structure)
    ↓
XGBoost Classifier (chống overfit)
    ↓
Structural SL (Swing Low/High thực tế, không phải ATR×N)
    ↓
Walk-Forward Validation (18 tháng, không look-ahead)
```

**Feature Importance từ XGBoost** (tự học, không áp đặt):
1. 🏆 `BullFVG` (Fair Value Gap Bullish): **32.1%**
2. 🥈 `BearFVG` (Fair Value Gap Bearish): **23.0%**
3. `Body_Pct`, `Dist_SwingLow`, `RSI_14`...

→ **FVG là signal quan trọng nhất**, không phải RSI hay Body như thường nghĩ.

---

## 🏗️ Kiến Trúc Pipeline

```
Raw M1 Data (710k nến, 2024-2025)
    │
    ▼ resample.py
M5 OHLCV (142k nến)
    │
    ▼ smc_features.py
SMC Features:
  ├── Swing High / Swing Low (pivot detection)
  ├── Market Structure (HH/HL vs LH/LL)
  ├── Order Block (Bullish OB / Bearish OB)
  ├── Fair Value Gap / Imbalance
  └── Distance features (normalize theo ATR)
    │
    ▼ signal_engine_v2.py
XGBoost Classifier (K=200, depth=4, anti-overfit)
    │
    ├── Confidence >= 60% + Active Session (London/NY)
    ├── Structural SL (Swing Low/High gần nhất, 2-5 pip)
    └── TP = Entry + SL_dist × 1.5 (R:R 1:1.5)
    │
    ▼ walk_forward.py
Walk-Forward 18 tháng: EV+0.183R, 14/18 tháng dương
    │
    ▼ monte_carlo.py
Monte Carlo 10k sims: 82% Pass Rate (Risk 1%)
```

---

## 📁 Cấu Trúc Project

```
├── python_proto/
│   ├── smc_features.py        # SMC Feature Engineering (OB, FVG, Swing)
│   ├── signal_engine_v2.py    # XGBoost Signal Engine chính
│   ├── walk_forward.py        # Walk-Forward Backtest 18 tháng
│   ├── monte_carlo.py         # Monte Carlo Prop Firm Simulator v3
│   ├── resample.py            # Resample M1 → M5/M15/H1
│   ├── grid_search_sltp.py    # Grid Search SL/TP Factor tối ưu
│   └── threshold_search.py    # Grid Search Confidence Threshold
├── docs/
│   ├── BRIEF.md
│   └── SPECS.md
├── plans/
│   └── 260312-pine-knn/
│       └── plan.md
└── README.md
```

---

## 🚀 Quick Start

### 1. Chuẩn Bị Data
Export M1 OHLCV từ MT5 (Nhấp phải → History Center → Export) cho XAUUSD:
```
XAUUSD_2024_all.csv  # Đặt ở thư mục gốc
XAUUSD_2025_all.csv  # Format: Date,Time,Open,High,Low,Close,Volume
```

### 2. Cài Đặt
```bash
pip install pandas numpy scikit-learn xgboost hmmlearn
```

### 3. Chạy Full Pipeline
```bash
# Bước 1: Xem Signal Engine chính (train + backtest)
python python_proto/signal_engine_v2.py

# Bước 2: Walk-Forward 18 tháng (validate không overfit)
python python_proto/walk_forward.py

# Bước 3: Monte Carlo Prop Firm Probability
python python_proto/monte_carlo.py
```

### 4. Sample Output
```
✅ PASS QUỸ tại lệnh 46 | Equity: $109,000

📊 XGBoost + SMC Signal Engine — M5 XAUUSD
  Số lệnh : 46 | Win: 22 | Loss: 24
  Winrate  : 47.8%
  EV/lệnh  : +0.196R ✅ EV DƯƠNG!
  SL       : Structural (Swing Low/High, 2-5 pip)

📝 Signal Mẫu:
  LONG   Entry: 3472.07 | SL: 3469.18 | TP: 3476.40 | Conf: 68.3% → WIN
  SHORT  Entry: 3482.65 | SL: 3484.70 | TP: 3479.57 | Conf: 71.2% → LOSS
  LONG   Entry: 3551.38 | SL: 3548.56 | TP: 3555.62 | Conf: 64.7% → WIN
```

---

## 🔬 Cải Tiến So Với Repo Tương Tự

| | Repo KNN Thông Thường | **Repo Này** |
|--|--|--|
| Model | KNN | **XGBoost + anti-overfit** |
| Features | RSI, MACD, BB | **SMC: OB, FVG, Swing** |
| SL placement | ATR × N (cứng) | **Structural (Swing Low/High)** |
| Validation | Train/Test cố định | **Walk-Forward 18 tháng** |
| Prop Firm | Không có | **Monte Carlo 10k sims** |
| EV | Âm (thực chiến) | **+0.183R Dương** |

---

## 🗺️ Roadmap

- [x] KNN Prototype (M1 XAUUSD)
- [x] Monte Carlo Prop Firm Simulator v3
- [x] SMC Feature Engineering
- [x] XGBoost Signal Engine v2
- [x] Walk-Forward Validation 18 tháng
- [ ] HMM Market Regime Detection (lọc tháng xấu)
- [ ] MT5 EA (MQL5) - Live Signal
- [ ] Tự động filter tháng Choppy (News calendar)

---

## ⚠️ Disclaimer

> Đây là **nghiên cứu và tham khảo**. Trading có rủi ro. Kết quả backtest **không đảm bảo** lợi nhuận trong tương lai.
> Luôn quản lý rủi ro và chỉ dùng vốn có thể chấp nhận mất.

---

## 👨‍💻 Tác Giả

**Phạm Phú Nguyễn Hưng** — Freelancer, Trader nghiệp dư, AI Automation Builder

Dự án này được xây dựng trong quá trình nghiên cứu chiến lược trading cho Prop Firm, kết hợp Machine Learning với Smart Money Concepts để tìm edge thực sự trên XAUUSD M5.

---

## 🤝 Bạn Muốn Hệ Thống Trading AI Tương Tự?

### Bạn cần → Chúng tôi đã làm ✅

| Bạn Cần | Chúng Tôi Đã Làm |
|---------|-----------------|
| Bot dự đoán nến XAUUSD/BTC | ✅ XGBoost + SMC Signal Engine |
| Backtest thực chiến Prop Firm | ✅ Walk-Forward + Monte Carlo |
| EA tự động trade MT5 | ✅ MQL5 EA với risk management |
| Dashboard tín hiệu real-time | ✅ Python + Telegram Alert |
| AI Automation cho doanh nghiệp | ✅ N8N + Agent workflows |

### 🚀 Liên Hệ Ngay

<div align="center">

[![Demo](https://img.shields.io/badge/🌐_Yêu_Cầu_Demo-comarai.com-purple?style=for-the-badge)](https://comarai.com)
[![Zalo](https://img.shields.io/badge/💬_Zalo-0834422439-blue?style=for-the-badge)](https://zalo.me/0834422439)
[![Email](https://img.shields.io/badge/📧_Email-hungphamphunguyen%40gmail.com-red?style=for-the-badge)](mailto:hungphamphunguyen@gmail.com)

</div>

---

### 🤖 Về Comarai — *Companion for Marketing & AI Automation*

> *"Thay vì thuê 5 nhân viên, tôi chạy 4 AI agent 24/7 với chi phí thấp hơn 90%."*
> — Nguyễn Hưng, Founder Comarai

**Đội ngũ AI của chúng tôi:**
- 🤖 **Em Sale** — Tự động tìm kiếm & outreach khách hàng B2B
- ✍️ **Em Content** — Tạo nội dung marketing đa kênh
- 📣 **Em Marketing** — Tối ưu chiến dịch & phân tích ROI
- 📈 **Em Trade** — Signal trading & risk management tự động

**[comarai.com](https://comarai.com)** | **[github.com/hungpixi](https://github.com/hungpixi)**
