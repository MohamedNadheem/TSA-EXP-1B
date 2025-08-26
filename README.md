# TSA-EXP-1B
Ex.No: 1B CONVERSION OF NON STATIONARY TO STATIONARY DATA
Date:
AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data

ALGORITHM:
Import the required packages like pandas and numpy
Read the data using the pandas
Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
Display the overall results.
PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Read dataset
data = pd.read_csv("asiacup.csv")

# Fill missing strike rates (forward fill)
data['Avg Bat Strike Rate'] = data['Avg Bat Strike Rate'].fillna(method='ffill')

# Aggregate average strike rate per year
data = data.groupby('Year')['Avg Bat Strike Rate'].mean().reset_index()

# Convert Year to datetime (add month/day so seasonal_decompose works)
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Set Year as index
data.set_index('Year', inplace=True)

# ===== Regular Differencing =====
data['strike_diff'] = data['Avg Bat Strike Rate'] - data['Avg Bat Strike Rate'].shift(1)

# ===== Seasonal Adjustment =====
result = seasonal_decompose(data['Avg Bat Strike Rate'], model='additive', period=1)
data['strike_sea_diff'] = result.resid

# ===== Log Transformation =====
data['strike_log'] = np.log(data['Avg Bat Strike Rate'])

# ===== Log Differencing =====
data['strike_log_diff'] = data['strike_log'] - data['strike_log'].shift(1)

# ===== Log Seasonal Differencing =====
result_log = seasonal_decompose(data['strike_log_diff'].dropna(), model='additive', period=1)
data['strike_log_seasonal_diff'] = result_log.resid

# ===== PLOTTING =====
plt.figure(figsize=(16, 16))

# 1. Original Data
plt.subplot(6, 1, 1)
plt.plot(data['Avg Bat Strike Rate'], label='Original')
plt.legend(loc='best')
plt.title('Original Data')
plt.xlabel('Year')
plt.ylabel('Avg Bat Strike Rate')

# 2. Regular Differencing
plt.subplot(6, 1, 2)
plt.plot(data['strike_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Year')
plt.ylabel('Diff(Avg Bat Strike Rate)')

# 3. Seasonal Adjustment
plt.subplot(6, 1, 3)
plt.plot(data['strike_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('Year')
plt.ylabel('Seasonally Adjusted Avg Bat Strike Rate')

# 4. Log Transformation
plt.subplot(6, 1, 4)
plt.plot(data['strike_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Year')
plt.ylabel('Log(Avg Bat Strike Rate)')

# 5. Log Transformation + Regular Differencing
plt.subplot(6, 1, 5)
plt.plot(data['strike_log_diff'], label='Log Transformation + Regular Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing')
plt.xlabel('Year')
plt.ylabel('RDiff(Log(Avg Bat Strike Rate))')

# 6. Log Transformation + Regular + Seasonal Differencing
plt.subplot(6, 1, 6)
plt.plot(data['strike_log_seasonal_diff'], label='Log Transformation + Regular Diff + Seasonal Diff')
plt.legend(loc='best')
plt.title('Log + Regular Diff + Seasonal Diff')
plt.xlabel('Year')
plt.ylabel('SDiff(RDiff(Log(Avg Bat Strike Rate)))')

plt.tight_layout()
plt.show()
```

OUTPUT:
REGULAR DIFFERENCING:

SEASONAL ADJUSTMENT:

LOG TRANSFORMATION:

RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger data.
