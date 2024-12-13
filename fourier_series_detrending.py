from scipy.signal import detrend
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with periodic trend
x = np.linspace(0, 10, 500)
trend = 0.5 * x
periodic = np.sin(2 * np.pi * x)
data = trend + periodic + np.random.normal(0, 0.1, len(x))

# Detrend the data
detrended = detrend(data)

# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(x, data, label="Original Data", alpha=0.6)
plt.plot(x, detrended, label="Detrended Data", alpha=0.9)
plt.legend()
plt.show()
