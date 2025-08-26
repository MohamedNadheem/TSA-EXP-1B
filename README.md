# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 26.08.2025
# Name: Mohamed Nadheem N

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:

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

### OUTPUT:


ORIGINAL DATA
<img width="1234" height="204" alt="Screenshot 2025-08-26 121512" src="https://github.com/user-attachments/assets/506d0572-d473-4e6e-994c-67b0c4a2c30f" />





REGULAR DIFFERENCING
<img width="1232" height="187" alt="image" src="https://github.com/user-attachments/assets/4890bb20-a936-4ddc-b482-5bc836f474aa" />





SEASONAL ADJUSTMENT
<img width="1275" height="210" alt="image" src="https://github.com/user-attachments/assets/0faf398b-5945-4d98-96a8-a6a01b61ecac" />





LOG TRANSFORMATION
<img width="1262" height="214" alt="Screenshot 2025-08-26 122216" src="https://github.com/user-attachments/assets/9197a749-eaa3-4a9d-8a5e-6cbd9d447cc8" />






LOG TRANSFORMATION AND REGULAR DIFFERENCING
<img width="1238" height="208" alt="image" src="https://github.com/user-attachments/assets/8e979d9f-6daf-4d39-86b7-d369283e387d" />






LOG + REGULAR DIFF + SEASONAL DIFF
<img width="1255" height="207" alt="image" src="https://github.com/user-attachments/assets/c0a02367-3130-45c1-b1fb-98f03160a0cd" />






### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
