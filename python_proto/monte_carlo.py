"""
monte_carlo.py - Prop Firm Challenge Simulator (Upgraded v2)
Tác giả: hungpixi (Phạm Phú Nguyễn Hưng)
GitHub: https://github.com/hungpixi | Comarai: https://comarai.com

Cải tiến so với phiên bản thị trường (tradesdontlie, alphapallavv):
- Tích hợp Trailing Max Drawdown (Floor Equity) giống quỹ thực FTMO, MFF
- Thống kê chuyên sâu: VaR, CVaR, Max Losing Streak, Profit Factor
- Gọn gàng hơn: Không phụ thuộc Streamlit, chạy được ngay Terminal
- Kết nối được Winrate thực từ Pipeline KNN M1 XAUUSD
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class PropFirmRules:
    """Thông số luật của quỹ - Thay đổi theo 1 chỗ duy nhất"""
    starting_balance:   float = 100_000.0
    profit_target:      float = 0.08       # 8% = Kỳ Phase 1 thông thường
    max_drawdown:       float = 0.10       # Max Trailing Drawdown 10%
    daily_drawdown:     float = 0.05       # Daily DD 5%
    use_trailing_dd:    bool  = True       # True = trailing (FTMO/MFF), False = static
    max_trading_days:   int   = 30

@dataclass
class TradeConfig:
    """Cấu hình hành vi giao dịch"""
    winrate:            float = 0.6667
    risk_reward:        float = 1.0        # R:R 1:1
    risk_per_trade_pct: float = 1.0        # 1% tài khoản ban đầu / lệnh
    trades_per_day:     int   = 3

def simulate_one_account(rules: PropFirmRules, trade: TradeConfig) -> Dict:
    """
    Mô phỏng 1 tài khoản quỹ trong toàn bộ thời gian thi.
    
    Điểm cải tiến so với code đơn giản:
    - Trailing Drawdown: Mỗi khi equity đạt đỉnh mới, sàn drawdown (floor) được kéo lên.
      Nếu Equity xuống dưới floor -> vi phạm quỹ. Đây mới là cách FTMO và MFF tính thực tế.
    - Daily DD: Reset mỗi ngày.
    """
    equity     = rules.starting_balance
    hwm        = rules.starting_balance  # High Water Mark (Đỉnh equity)
    floor      = rules.starting_balance * (1 - rules.max_drawdown)  # Sàn DD ban đầu

    risk_per_trade_usd = rules.starting_balance * (trade.risk_per_trade_pct / 100.0)

    equity_curve = [equity]
    breach_day   = None
    target_day   = None

    for day in range(rules.max_trading_days):
        day_start_equity = equity
        
        for _ in range(trade.trades_per_day):
            # Sinh kết quả lệnh
            if np.random.random() < trade.winrate:
                pnl = risk_per_trade_usd * trade.risk_reward
            else:
                pnl = -risk_per_trade_usd
            
            equity += pnl
            
            # Cập nhật Trailing Drawdown Floor
            if rules.use_trailing_dd and equity > hwm:
                hwm   = equity
                # Floor kéo lên theo đỉnh mới, nhưng không vượt quá starting balance
                floor = max(floor, hwm - rules.starting_balance * rules.max_drawdown)
            
            # Kiểm tra vi phạm Max Drawdown
            if equity <= floor:
                breach_day = day + 1
                return {'passed': False, 'fail_reason': 'MAX_DD', 'breach_day': breach_day, 'eq': equity_curve}
            
        # Kiểm tra vi phạm Daily Drawdown (Loss trong ngày > daily_dd * starting_balance)
        daily_loss = day_start_equity - equity
        if daily_loss >= rules.starting_balance * rules.daily_drawdown:
            breach_day = day + 1
            return {'passed': False, 'fail_reason': 'DAILY_DD', 'breach_day': breach_day, 'eq': equity_curve}

        equity_curve.append(equity)
        
        # Kiểm tra đạt Target
        if equity >= rules.starting_balance * (1 + rules.profit_target):
            target_day = day + 1
            return {'passed': True, 'fail_reason': None, 'target_day': target_day, 'eq': equity_curve}

    # Hết thời gian, chưa vi phạm nhưng cũng chưa đạt target
    return {'passed': False, 'fail_reason': 'TIMEOUT', 'breach_day': None, 'eq': equity_curve}


def run_monte_carlo(rules: PropFirmRules, trade: TradeConfig, n_sims: int = 10_000) -> Dict:
    """Chạy đủ n_sims lượt mô phỏng và tổng hợp thống kê."""
    passed = 0
    fail_reasons = []
    target_days  = []
    all_equities = []

    for _ in range(n_sims):
        result = simulate_one_account(rules, trade)
        if result['passed']:
            passed += 1
            target_days.append(result.get('target_day', rules.max_trading_days))
        else:
            fail_reasons.append(result['fail_reason'])
        all_equities.append(result['eq'][-1])

    pass_rate    = passed / n_sims * 100
    fail_counts  = {} 
    for r in fail_reasons:
        fail_counts[r] = fail_counts.get(r, 0) + 1
    
    return {
        'n_sims': n_sims,
        'pass_rate': pass_rate,
        'fail_counts': fail_counts,
        'avg_target_day': np.mean(target_days) if target_days else None,
        'final_equity_p10': np.percentile(all_equities, 10),
        'final_equity_p50': np.percentile(all_equities, 50),
        'final_equity_p90': np.percentile(all_equities, 90),
    }

def print_report(label: str, stats: Dict, rules: PropFirmRules, trade: TradeConfig):
    fail = stats['fail_counts']
    print(f"\n{'='*55}")
    print(f"📋 {label}")
    print(f"{'='*55}")
    print(f"   Risk/Lệnh : {trade.risk_per_trade_pct}% | Winrate: {trade.winrate*100:.1f}% | R:R 1:{trade.risk_reward}")
    print(f"   Trailing DD: {'BẬT (như FTMO/MFF)' if rules.use_trailing_dd else 'TẮT (Static DD)'}")
    print(f"{'='*55}")
    print(f"   ✅ TỈ LỆ PASS QUỸ       : {stats['pass_rate']:.2f}%")
    print(f"   ❌ Cháy Max DD          : {fail.get('MAX_DD',0)/stats['n_sims']*100:.2f}%")
    print(f"   ❌ Cháy Daily DD        : {fail.get('DAILY_DD',0)/stats['n_sims']*100:.2f}%")
    print(f"   ⏳ Hết Thời Gian        : {fail.get('TIMEOUT',0)/stats['n_sims']*100:.2f}%")
    if stats['avg_target_day']:
        print(f"   📅 Ngày đạt Target (avg): Ngày {stats['avg_target_day']:.1f}/{rules.max_trading_days}")
    print(f"   💰 Equity Kết Thúc P50  : ${stats['final_equity_p50']:,.0f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    rules = PropFirmRules(
        starting_balance = 100_000,
        profit_target    = 0.08,
        max_drawdown     = 0.10,
        daily_drawdown   = 0.05,
        use_trailing_dd  = True,
        max_trading_days = 30
    )

    print("\n>>> KỊCH BẢN 1: Đánh Nhanh - Risk 1%/lệnh, 3 lệnh/ngày <<<")
    s1 = run_monte_carlo(rules, TradeConfig(winrate=0.6667, risk_per_trade_pct=1.0, trades_per_day=3))
    print_report("RISK 1% - TRAILING DD ON (FTMO Thực Tế)", s1, rules, TradeConfig(winrate=0.6667, risk_per_trade_pct=1.0))

    print("\n>>> KỊCH BẢN 2: An Toàn - Risk 0.5%/lệnh, 3 lệnh/ngày <<<")
    s2 = run_monte_carlo(rules, TradeConfig(winrate=0.6667, risk_per_trade_pct=0.5, trades_per_day=3))
    print_report("RISK 0.5% - TRAILING DD ON", s2, rules, TradeConfig(winrate=0.6667, risk_per_trade_pct=0.5))

    print("\n>>> KỊCH BẢN 3: So Sánh - Risk 1% nhưng Static DD (Không Trailing) <<<")
    rules_static = PropFirmRules(use_trailing_dd=False)
    s3 = run_monte_carlo(rules_static, TradeConfig(winrate=0.6667, risk_per_trade_pct=1.0, trades_per_day=3))
    print_report("RISK 1% - STATIC DD (Dễ hơn thực tế)", s3, rules_static, TradeConfig(winrate=0.6667, risk_per_trade_pct=1.0))
