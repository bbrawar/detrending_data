import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate cyclic data
days = 30
data = np.sin(2 * np.pi * np.linspace(0, 1, days)) + np.random.normal(0, 0.2, days)
data = np.tile(data, 12)  # Repeat data for 12 months

# Reshape into a matrix for each cycle
data_matrix = np.reshape(data, (-1, days))
average_cycle = np.mean(data_matrix, axis=0)

# Detrend data by removing the average cycle
detrended_data = data - np.tile(average_cycle, 12)

# Plot the original and detrended data
plt.figure(figsize=(10, 5))
plt.plot(data, label="Original Data")
plt.plot(np.tile(average_cycle, 12), label="Average Cycle", linestyle="--")
plt.plot(detrended_data, label="Detrended Data", alpha=0.7)
plt.legend()
plt.show()
