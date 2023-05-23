import numpy as np

def detect_constant_price(stock_prices, duration):
    # Calculate the difference between each price and the next one
    differences = np.diff(stock_prices)

    # Create an array where the value is True when the price is the same as the next day
    unchanged = differences == 0

    # Create an array to count the number of continuous unchanged prices
    counts = np.cumsum(unchanged)
    counts[~unchanged] = 0

    # Check if there's a period where the price remained constant for the required duration
    return np.max(counts) >= duration

