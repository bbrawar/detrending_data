import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def rolling_barrel_detrending(time, tec, window_radius):
    """
    Perform a rolling-barrel detrending algorithm on TEC data.

    Parameters:
        time (array): The time axis data (e.g., in hours).
        tec (array): TEC signal data (in TECU).
        window_radius (float): Half-width of the rolling barrel window (in time units).

    Returns:
        trend (array): The inferred trend of the TEC data.
        detrended (array): The detrended TEC signal.
    """
    trend = np.zeros_like(tec)
    n = len(tec)

    # Calculate the rolling trend using a window centered on each point
    for i in range(n):
        # Define the window limits
        start_idx = max(0, i - window_radius)
        end_idx = min(n, i + window_radius + 1)

        # Fit a straight line to the points in the window (trend approximation)
        window_time = time[start_idx:end_idx]
        window_tec = tec[start_idx:end_idx]
        poly_coeff = np.polyfit(window_time, window_tec, 1)  # Linear fit (degree = 1)
        trend[i] = np.polyval(poly_coeff, time[i])

    # Calculate the detrended signal
    detrended = tec - trend
    return trend, detrended

# Generate synthetic TEC data with trends and irregularities
np.random.seed(42)
time = np.linspace(0, 6, 500)  # Time in hours
trend_true = 50 - 8 * time + savgol_filter(np.random.normal(0, 0.5, len(time)), 51, 3)
irregularities = np.sin(6 * np.pi * time) * (time > 2) * (time < 5)
tec = trend_true + irregularities + np.random.normal(0, 0.8, len(time))

# Apply rolling barrel detrending
window_radius = 25  # Radius in index units (adjust based on time resolution)
trend, detrended = rolling_barrel_detrending(time, tec, window_radius)

# Plotting the results
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(time, tec, label="Original TEC", color="blue")
plt.plot(time, trend, label="Inferred Trend", color="red")
plt.xlabel("Time (hours)")
plt.ylabel("TEC (TECU)")
plt.title("Original TEC and Inferred Trend")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, detrended, label="Detrended TEC", color="green")
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Time (hours)")
plt.ylabel("Detrended TEC (TECU)")
plt.title("Detrended TEC Signal")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, np.abs(np.gradient(detrended, time)), label="|dTEC/dt|", color="purple")
plt.xlabel("Time (hours)")
plt.ylabel("|dTEC/dt| (TECU/hr)")
plt.title("Gradient of Detrended TEC Signal")
plt.legend()

plt.tight_layout()
plt.show()
