import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data with a trend and periodic component
x = np.linspace(0, 10, 500)
trend = 0.5 * x
periodic = np.sin(2 * np.pi * x)
data = trend + periodic + np.random.normal(0, 0.1, len(x))

# Apply a rolling average to detrend
window_size = 50
rolling_avg = pd.Series(data).rolling(window=window_size, center=True).mean()

# Detrended data
detrended = data - rolling_avg

# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(x, data, label="Original Data", alpha=0.6)
plt.plot(x, rolling_avg, label="Rolling Average", linestyle='--')
plt.plot(x, detrended, label="Detrended Data", alpha=0.9)
plt.legend()
plt.show()
