import logging
import time
from brokers.capital import CapitalBroker
from strategy import atr_stop_distance
from config import (
    ATR_STOP_MULT,
    ATR_STOP_WINDOW,
    STOP_LOSS_PCT,
    TAKE_PROFIT_MULT,
    TAKE_PROFIT_PCT,
)

# Enable basic logging to see any errors
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("Connecting to Capital.com...")
    broker = CapitalBroker()
    
    # Check buying power
    bp = broker.get_buying_power()
    print(f"Current Buying Power: ${bp:,.2f}")
    
    if bp < 100:
        print("Not enough money to safely place a test order!")
        return

    # A very tiny quantity just to test (0.001 BTC is roughly ~$70)
    test_qty = 0.001
    symbol = "BTC/USD"
    
    # Fetch recent bars to calculate SL/TP
    print(f"Fetching data for {symbol}...")
    bars = broker.get_recent_bars(symbol, limit=30)
    latest_price = float(bars["c"].iloc[-1])
    
    # Calculate SL/TP levels
    atr = atr_stop_distance(bars["h"], bars["l"], bars["c"], window=ATR_STOP_WINDOW, multiplier=1.0)
    sl_dist = max(atr * ATR_STOP_MULT, latest_price * STOP_LOSS_PCT)
    tp_dist = max(atr * TAKE_PROFIT_MULT, latest_price * TAKE_PROFIT_PCT)
    
    sl_price = round(latest_price - sl_dist, 2)
    tp_price = round(latest_price + tp_dist, 2)

    print(f"\nAttempting to BUY {test_qty} {symbol} @ ~{latest_price:.2f}...")
    print(f"Proposed SL: {sl_price:.2f}")
    print(f"Proposed TP: {tp_price:.2f}")

    try:
        broker.submit_buy(symbol, test_qty, sl=sl_price, tp=tp_price)
        print("SUCCESS: Market BUY order submitted without errors!")
        
        print("Waiting for order to fill...")
        for _ in range(5):
            time.sleep(2)
            broker.flush_position_cache()
            pos_qty = broker.get_position_qty(symbol)
            if pos_qty > 0:
                print(f"Current held quantity of {symbol}: {pos_qty}")
                break
        else:
            print(f"Order for {symbol} not filled yet or position not found after 10 seconds.")
        
    except Exception as e:
        print(f"ERROR submitting order: {e}")

if __name__ == "__main__":
    main()
