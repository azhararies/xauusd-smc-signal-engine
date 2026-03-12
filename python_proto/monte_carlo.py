import numpy as np
import pandas as pd

def run_monte_carlo_prop_firm(
    winrate=0.6667, 
    risk_reward=1.0, 
    risk_per_trade_pct=1.0, 
    profit_target_pct=8.0, 
    max_drawdown_pct=10.0,
    daily_drawdown_pct=5.0, # Giả định đơn giản: DD liên tục mà không có lệnh win ngắt quãng trong ngày
    trades_per_day=5,
    max_trading_days=30,
    num_simulations=10000
):
    """
    Monte Carlo Simulation cho Prop Firm Challenge.
    Mô phỏng 10,000 tài khoản quỹ với thông số Winrate và R:R cho trước.
    """
    print(f"[*] Chạy Monte Carlo Simulation ({num_simulations} tài khoản)...")
    print(f"    - Winrate: {winrate * 100:.2f}% | R:R: 1:{risk_reward}")
    print(f"    - Risk/Lệnh: {risk_per_trade_pct}% | Target Pass: {profit_target_pct}%")
    
    total_trades_limit = trades_per_day * max_trading_days
    
    # Kết quả của mỗi lượt mô phỏng
    passed_count = 0
    failed_dd_count = 0
    failed_time_count = 0
    
    # Sinh ma trận ngẫu nhiên cho tất cả các mô phỏng (Vectorized cho tốc độ cực nhanh)
    # 1 = Win, -1 = Loss
    random_probs = np.random.rand(num_simulations, total_trades_limit)
    trade_results = np.where(random_probs <= winrate, risk_per_trade_pct * risk_reward, -risk_per_trade_pct)
    
    # Tính Equity Curve dạng tích lũy (Cumulative Sum)
    # Equity ban đầu = 100% -> profit cộng dồn
    equity_curves = np.cumsum(trade_results, axis=1)
    
    for i in range(num_simulations):
        curve = equity_curves[i]
        
        # Biến theo dõi Max DD
        running_max = 0
        passed = False
        failed = False
        
        for j, current_profit in enumerate(curve):
            # Cập nhật High Watermark (Kỷ lục equity)
            if current_profit > running_max:
                running_max = current_profit
                
            # Tính Drawdown từ đỉnh
            current_dd = running_max - current_profit
            
            # Kiểm tra Luật Quỹ
            if current_dd >= max_drawdown_pct:
                failed_dd_count += 1
                failed = True
                break
                
            # Đơn giản hóa Daily DD thành chuỗi thua liên tiếp tối đa trong 1 ngày
            # Trên thực tế, có thể cần tracking theo ngày. Ở đây mình dùng threshold sụt giảm gắt.
            if current_profit <= (-daily_drawdown_pct):
                 failed_dd_count += 1
                 failed = True
                 break

            # Kiểm tra Target (Pass)
            if current_profit >= profit_target_pct:
                passed_count += 1
                passed = True
                break
                
        if not passed and not failed:
             # Nết hết thời gian mà chưa chạm target cũng chưa cháy
             failed_time_count += 1

    pass_rate = (passed_count / num_simulations) * 100
    fail_dd_rate = (failed_dd_count / num_simulations) * 100
    fail_time_rate = (failed_time_count / num_simulations) * 100
    
    print("\n" + "="*50)
    print("📊 MÔ PHỎNG PROP FIRM - KẾT QUẢ MONTE CARLO")
    print("="*50)
    print(f"Tổng số tài khoản mô phỏng: {num_simulations:,}")
    print(f"✅ TỈ LỆ PASS QUỸ:      {pass_rate:.2f}%")
    print(f"❌ Tỉ lệ Cháy (Max/Daily DD): {fail_dd_rate:.2f}%")
    print(f"⏳ Tỉ lệ Hết Thời Gian:        {fail_time_rate:.2f}%")
    print("="*50)
    
    # Khuyến nghị tự động
    if pass_rate < 50:
        print("⚠️ KHUYẾN NGHỊ: Tỉ lệ Pass thấp. Cần giảm Risk per trade hoặc tối ưu R:R lên 1:1.5")
    elif fail_dd_rate > 30:
        print("⚠️ KHUYẾN NGHỊ: Drawdown căng rủi ro! Hãy giảm Risk/Lệnh xuống 0.5% hoặc 0.25%.")
    else:
        print("🎯 KHUYẾN NGHỊ: Chiến lược cực kỳ ổn định. Có thể tự tin thi quỹ với cấu hình này!")

if __name__ == "__main__":
    # Test thông số lý thuyết từ kết quả Backtest của anh Hưng
    # Winrate ròng đợt test 40k nến M1 là 66.67% (26W/13L)
    # Assume: Đánh quỹ FTMO Phase 1 (Target 8%, Max DD 10%, Risk 1%/lệnh)
    run_monte_carlo_prop_firm(
        winrate=0.6667, 
        risk_reward=1.0, 
        risk_per_trade_pct=1.0, # Đánh 1% tài khoản 1 lệnh
        max_trading_days=30,
        trades_per_day=3
    )
    
    # Test thử hạ Risk xuống 0.5%
    print("\n\n>>> KỊCH BẢN 2: HẠ RISK XUỐNG 0.5% (Tâm lý thoải mái hơn) <<<")
    run_monte_carlo_prop_firm(
        winrate=0.6667, 
        risk_reward=1.0, 
        risk_per_trade_pct=0.5, # Đánh 0.5% tài khoản 1 lệnh
        max_trading_days=30,
        trades_per_day=3
    )
