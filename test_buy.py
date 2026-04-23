import logging



# Enable basic logging to see any errors
logging.basicConfig(level=logging.INFO)

def main():
    print("Connecting to Capital.com...")
    from brokers.capital import CapitalBroker
    broker = CapitalBroker()
    
    # Check buying power
    bp = broker.get_buying_power()
    print(f"Current Buying Power: ${bp:,.2f}")
    
    if bp < 100:
        print("Not enough paper money to safely place a test order!")
        return

    # A very tiny quantity just to test (0.001 BTC is roughly ~$70)
    test_qty = 0.001
    symbol = "BTC/USD"
    
    print(f"\nAttempting to buy {test_qty} {symbol}...")
    try:
        broker.submit_buy(symbol, test_qty)
        print("SUCCESS: Market BUY order submitted without errors!")
        
        # Let's see if it filled
        import time
        print("Waiting for order to fill...")
        
        for _ in range(5):
            time.sleep(2)  # pause for Capital.com to fill the paper market order
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
