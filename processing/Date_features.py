# Date_features.py
import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('./data/SeeqExport_LOS_raw.csv', sep=';', parse_dates=['Date-Time'])  # Replace with your file path

# Extract the month and the year as additional features
df['month'] = df['Date-Time'].dt.month 
df['year'] = df['Date-Time'].dt.year.astype(str).str[-2:] 

# Set date as index
df.set_index('Date-Time', inplace=True)
print(df.head())
print(df.tail())

# Remove duplicate Date-Times (sanity check)
duplicates = df.index.duplicated()
print(df[duplicates])  # Display duplicate rows
print(len(df[duplicates]))  # Display duplicate rows

data = df[~df.index.duplicated()] # Remove
print(len(data))

# Resample to hourly frequency
data = data.asfreq(freq='h', method='bfill')  
print(len(data))

print(data.head())
print(data.tail())
shape = data.shape
print(shape)

# Save
data.to_csv('./data/LOS_CLEANED_interp.csv', index=False)