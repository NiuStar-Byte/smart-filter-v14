# pecnew_logic.py

# Function for current PEC logic (existing logic)
def process_PEC(data):
    print("Processing current PEC logic...")
    # Your PEC logic code goes here
    return "Current PEC processing complete."

# PECNew logic (P&L exits, time-based exits)
def process_PECNew(data):
    print("Processing PECNew logic...")
    
    # Step 1: Extract entry price, current price, and other relevant data
    entry_price = data['entry_price']  # Replace with actual data field
    current_price = data['current_price']  # Replace with current price field
    signal_type = data['signal_type']  # LONG or SHORT
    
    # Step 2: Define TP and SL based on signal type
    if signal_type == 'LONG':
        tp = entry_price * 1.30  # TP 30% for LONG
        sl = entry_price * 0.80  # SL 20% for LONG
        time_limit = 16  # 3mTF: 16 minutes for LONG
    elif signal_type == 'SHORT':
        tp = entry_price * 0.70  # TP 30% for SHORT (price should decrease)
        sl = entry_price * 1.20  # SL 20% for SHORT (price should increase)
        time_limit = 26  # 5mTF: 26 minutes for SHORT
    else:
        tp = sl = time_limit = None  # Handle any edge cases
    
    # Step 3: Check if TP or SL is hit within the timeframe
    if current_price >= tp:
        print(f"TP hit for {signal_type}! Exit position.")
        return f"Exit - TP hit for {signal_type}"
    elif current_price <= sl:
        print(f"SL hit for {signal_type}! Exit position.")
        return f"Exit - SL hit for {signal_type}"
    elif data['time_elapsed'] >= time_limit:
        print(f"Time limit reached for {signal_type}! Exit based on current price: {current_price}")
        return f"Exit - Time limit reached for {signal_type}, current price: {current_price}"

    return "No exit conditions met"

# PECNew backtest logic (simulating trades based on historical data)
def process_PECNew_backtest(data):
    print("Processing PECNew backtest...")
    
    # Step 1: Extract entry price and other relevant data
    entry_price = data['entry_price']  # Replace with historical data field
    current_price = data['current_price']  # Replace with historical price data field
    signal_type = data['signal_type']  # LONG or SHORT
    time_limit = 16 if signal_type == 'LONG' else 26  # Time limits for 3mTF or 5mTF
    
    # Implement backtesting logic
    for bar in historical_data:
        current_price = bar['close']  # Replace with actual close price from historical data
        
        # Check if TP or SL is hit during backtest
        if current_price >= entry_price * 1.30:
            print("Backtest TP hit!")
            return "Backtest Exit - TP hit"
        elif current_price <= entry_price * 0.80:
            print("Backtest SL hit!")
            return "Backtest Exit - SL hit"
        elif bar['time_elapsed'] >= time_limit:
            print("Backtest Time limit reached!")
            return f"Backtest Exit - Time limit reached, current price: {current_price}"
    
    return "Backtest: No exit conditions met"
