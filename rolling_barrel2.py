import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rolling_barrel_detrending_df(df, column, window_radius):
    """
    Perform rolling-barrel detrending on TEC data stored in a DataFrame.

    Parameters:
        df (DataFrame): The TEC data with timestamps as index.
        column (str): The column name containing TEC data.
        window_radius (int): Half-width of the rolling barrel window (in rows).

    Returns:
        DataFrame: A DataFrame with the trend and detrended TEC data.
    """
    tec = df[column].values
    trend = np.zeros_like(tec)
    n = len(tec)
    time = (df.index - df.index[0]).total_seconds() / 3600  # Convert index to hours

    for i in range(n):
        start_idx = max(0, i - window_radius)
        end_idx = min(n, i + window_radius + 1)
        window_time = time[start_idx:end_idx]
        window_tec = tec[start_idx:end_idx]
        if len(window_time) > 1:  # Fit linear trend only if enough data points exist
            poly_coeff = np.polyfit(window_time, window_tec, 1)
            trend[i] = np.polyval(poly_coeff, time[i])

    detrended = tec - trend
    result = df.copy()
    result['Trend'] = trend
    result['Detrended'] = detrended
    return result

# Example: Create a synthetic DataFrame
time_index = pd.date_range(start="2024-01-01 00:00:00", periods=500, freq="H")
tec_data = 50 - 8 * np.linspace(0, 6, 500) + np.random.normal(0, 1, 500)
df = pd.DataFrame({'TEC': tec_data}, index=time_index)

# Apply rolling barrel detrending
window_radius = 25  # Adjust based on the dataset frequency
df_result = rolling_barrel_detrending_df(df, column='TEC', window_radius=window_radius)

# Plotting
plt.figure(figsize=(14, 8))

# Original and Trend
plt.subplot(3, 1, 1)
plt.plot(df.index, df['TEC'], label='Original TEC', color='blue')
plt.plot(df_result.index, df_result['Trend'], label='Inferred Trend', color='red')
plt.xlabel("Timestamp")
plt.ylabel("TEC (TECU)")
plt.title("Original TEC and Inferred Trend")
plt.legend()

# Detrended Signal
plt.subplot(3, 1, 2)
plt.plot(df_result.index, df_result['Detrended'], label='Detrended TEC', color='green')
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Timestamp")
plt.ylabel("Detrended TEC (TECU)")
plt.title("Detrended TEC Signal")
plt.legend()

# Gradient of Detrended Signal
plt.subplot(3, 1, 3)
gradient = np.abs(np.gradient(df_result['Detrended'], np.arange(len(df_result))))
plt.plot(df_result.index, gradient, label="|dTEC/dt|", color="purple")
plt.xlabel("Timestamp")
plt.ylabel("|dTEC/dt| (TECU/hr)")
plt.title("Gradient of Detrended TEC Signal")
plt.legend()

plt.tight_layout()
plt.show()
