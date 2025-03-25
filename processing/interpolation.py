import pandas as pd

df = pd.read_csv('./data/LOS_raw/x.csv', parse_dates=['Date']) #Switch
df.set_index('Date', inplace=True)

# Aggregate duplicates (mean)
df = df.groupby(df.index).mean()

start_date = df.index.min()
end_date = df.index.max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='h')

# Reindex
df_reindexed = df.reindex(all_dates)

# Interpolate
df_reindexed['x'] = df_reindexed['x'].interpolate(method='time') # Switch x with appropriate measurement
df_reindexed.reset_index(inplace=True)
df_reindexed.rename(columns={'index': 'Date'}, inplace=True)

df_reindexed.to_csv('./data/processed/LOS_x_CLEANED_interp.csv', index=False)