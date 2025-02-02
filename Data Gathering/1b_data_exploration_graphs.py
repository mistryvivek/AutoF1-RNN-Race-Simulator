import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder

dataset2018 = pd.read_csv("dataset2018.csv", low_memory=False, index_col=0)
dataset2019 = pd.read_csv("dataset2019.csv", low_memory=False, index_col=0)
dataset2020 = pd.read_csv("dataset2020.csv", low_memory=False, index_col=0)
dataset2021 = pd.read_csv("dataset2021.csv", low_memory=False, index_col=0)
dataset2022 = pd.read_csv("dataset2022.csv", low_memory=False, index_col=0)
dataset2023 = pd.read_csv("dataset2023.csv", low_memory=False, index_col=0)
dataset2024 = pd.read_csv("dataset2024.csv", low_memory=False, index_col=0)

dataset2018.loc[:, 'LapTime'] = pd.to_timedelta(dataset2018['LapTime'], errors='coerce')
dataset2019.loc[:, 'LapTime'] = pd.to_timedelta(dataset2019['LapTime'], errors='coerce')
dataset2020.loc[:, 'LapTime'] = pd.to_timedelta(dataset2020['LapTime'], errors='coerce')
dataset2021.loc[:, 'LapTime'] = pd.to_timedelta(dataset2021['LapTime'], errors='coerce')
dataset2022.loc[:, 'LapTime'] = pd.to_timedelta(dataset2022['LapTime'], errors='coerce')
dataset2023.loc[:, 'LapTime'] = pd.to_timedelta(dataset2023['LapTime'], errors='coerce')
dataset2024.loc[:, 'LapTime'] = pd.to_timedelta(dataset2024['LapTime'], errors='coerce')

racesDataset2018 = dataset2018[dataset2018['Session'] == 'Race']
racesDataset2019 = dataset2019[dataset2019['Session'] == 'Race']
racesDataset2020 = dataset2020[dataset2020['Session'] == 'Race']
racesDataset2021 = dataset2021[dataset2021['Session'] == 'Race']
racesDataset2022 = dataset2022[dataset2022['Session'] == 'Race']
racesDataset2023 = dataset2023[dataset2023['Session'] == 'Race']
racesDataset2024 = dataset2024[dataset2024['Session'] == 'Race']

# Boxplot to look at low + high deg tracks.
location = {}
for dataset in [racesDataset2018, racesDataset2019, racesDataset2020, racesDataset2021, racesDataset2022, racesDataset2023, racesDataset2024]:
    for event in dataset['EventName'].unique():
        eventDataset = dataset[dataset['EventName'] == event]
        for driver in eventDataset['Driver'].unique():
            driverDataset = eventDataset[eventDataset['Driver'] == driver]
            for stint in driverDataset['Stint'].unique():
                stintDataset = driverDataset[(driverDataset['Stint'] == stint) & (driverDataset['PitOutTime'].isna()) & (driverDataset['PitInTime'].isna())].copy()
                if len(stintDataset) > 1 and not stintDataset['LapTime'].isna().any():
                    stintDataset.loc[:, 'LapTimeDiff'] = abs(stintDataset['LapTime'].diff().dt.total_seconds())
                    average_diff = stintDataset['LapTimeDiff'].mean()
                    if stintDataset.iloc[0]['Location'] in location.keys():
                        location[stintDataset.iloc[0]['Location']].append(average_diff)
                    else:
                        location[stintDataset.iloc[0]['Location']] = [average_diff]

iqr_dict = {
    location: np.percentile(values, 75) - np.percentile(values, 25)
    for location, values in location.items()
}

location = dict(sorted(location.items(), key=lambda x: sum(x[1]) / len(x[1])))
location_names = list(location.keys())  # X-axis labels
degradation_values = list(location.values())  # Y-axis data

# Create the box plot
plt.figure(figsize=(12, 6))
plt.boxplot(degradation_values, tick_labels=location_names, patch_artist=True, boxprops=dict(facecolor="lightblue"))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, fontsize=10, ha='right')

# Add grid for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Customize title and labels
plt.title('Tire Degradation by Location', fontsize=14, fontweight='bold')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Degradation Values', fontsize=12)

# Tighten layout
plt.tight_layout()

# Show the plot
plt.show()

# Correlation testing.
combined_dataset = pd.concat([dataset2018, dataset2019, dataset2020, dataset2021, dataset2022, dataset2023, dataset2024])
combined_dataset_corr_cols = combined_dataset[['LapTime', 'Driver', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Compound', 'TyreLife', 'FreshTyre', 'Team', 'TrackStatus', 'Brake', 'DRS', 'Status', 'DistanceToDriverAhead', 'DistanceToDriverBehind', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
       'WindDirection', 'WindSpeed', 'Country', 'Location']].copy()
combined_dataset_corr_cols = combined_dataset_corr_cols[combined_dataset_corr_cols['LapTime'].apply(lambda x: isinstance(x, pd.Timedelta))]
encoder = LabelEncoder()
# Encode compounds.
combined_dataset_corr_cols['Compound'] = encoder.fit_transform(combined_dataset_corr_cols['Compound']).astype(float)
combined_dataset_corr_cols['Driver'] = encoder.fit_transform(combined_dataset_corr_cols['Driver']).astype(float)
combined_dataset_corr_cols['Team'] = encoder.fit_transform(combined_dataset_corr_cols['Team']).astype(float)
combined_dataset_corr_cols['Status'] = encoder.fit_transform(combined_dataset_corr_cols['Status']).astype(float)
combined_dataset_corr_cols['Country'] = encoder.fit_transform(combined_dataset_corr_cols['Country']).astype(float)
combined_dataset_corr_cols['Location'] = encoder.fit_transform(combined_dataset_corr_cols['Location']).astype(float)
combined_dataset_corr_cols['LapTime'] = combined_dataset_corr_cols['LapTime'].apply(lambda x: x.total_seconds())
correlations = combined_dataset_corr_cols.corr(method='pearson')
print("Pearsons: ")
print(correlations['LapTime'])
correlations = combined_dataset_corr_cols.corr(method='spearman')
print("Spearman: ")
print(correlations['LapTime'])

# Look at the pit stop data.
pit_stop_df = pd.DataFrame()
combined_races_dataset = pd.concat([racesDataset2018, racesDataset2019, racesDataset2020, racesDataset2021, racesDataset2022, racesDataset2023, racesDataset2024])

for event in combined_races_dataset['EventName'].unique():
    dfEvent = combined_races_dataset[combined_races_dataset['EventName'] == event].copy()
    
    for year in dfEvent['Year'].unique():
        for driver in dfEvent['Driver'].unique():
            dfDriver = dfEvent[(dfEvent['Driver'] == driver) & (dfEvent['Year'] == year)].copy()
            
            if dfDriver.shape[0] > 1:
                dfDriver = dfDriver.sort_values(by='LapNumber').reset_index(drop=True)
                dfDriver['StintChange'] = dfDriver['Compound'].shift(-1).where(dfDriver['Stint'] != dfDriver['Stint'].shift(-1), "NO PIT")
                dfDriver = dfDriver[:-1]
                pit_stop_df = pd.concat([pit_stop_df, dfDriver], ignore_index=True)
                print(dfDriver['Compound'].value_counts())

pit_stop_corr_cols = pit_stop_df[['LapTime', 'Driver', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Compound', 'TyreLife', 'FreshTyre', 'Team', 'TrackStatus', 'Brake', 'DRS', 'Status', 'DistanceToDriverAhead', 'DistanceToDriverBehind', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
       'WindDirection', 'WindSpeed', 'Country', 'Location', 'StintChange']].copy()
pit_stop_corr_cols['Compound'] = encoder.fit_transform(pit_stop_corr_cols['Compound']).astype(float)
pit_stop_corr_cols['Driver'] = encoder.fit_transform(pit_stop_corr_cols['Driver']).astype(float)
pit_stop_corr_cols['Team'] = encoder.fit_transform(pit_stop_corr_cols['Team']).astype(float)
pit_stop_corr_cols['Status'] = encoder.fit_transform(pit_stop_corr_cols['Status']).astype(float)
pit_stop_corr_cols['Country'] = encoder.fit_transform(pit_stop_corr_cols['Country']).astype(float)
pit_stop_corr_cols['Location'] = encoder.fit_transform(pit_stop_corr_cols['Location']).astype(float)
pit_stop_corr_cols['StintChange'] = encoder.fit_transform(pit_stop_corr_cols['StintChange']).astype(float)
pit_stop_corr_cols['LapTime'] = pit_stop_corr_cols['LapTime'].apply(lambda x: x.total_seconds())
correlations = pit_stop_corr_cols.corr(method='pearson')
print("Pearsons (PIT): ")
print(correlations['StintChange'])
correlations = pit_stop_corr_cols.corr(method='spearman')
print("Spearman (PIT): ")
print(correlations['StintChange'])

# Plot effect of under/overcut.
DNFS = ['R', 'D', 'E', 'W', 'F', 'N', 'U']
pit_stop_df['ClassifiedPosition'].fillna('U')
# 24 places on the grid is the highest, if they DNF'ed the position change will be low enough to return a negative value.
pit_stop_df['ClassifiedPosition'] = pit_stop_df['ClassifiedPosition'].apply(
    lambda x: 25.0 if x in DNFS else float(x)
)

pit_stop_df['PositionChange'] = pit_stop_df['GridPosition'] - pit_stop_df['ClassifiedPosition']
pit_stop_df['PositionGain'] = pit_stop_df['PositionChange'] > 0

pit_stop_laps = pit_stop_df[pit_stop_df['StintChange'] != "NO PIT"]
pit_stop_laps = pit_stop_df.dropna(subset=['DistanceToDriverAhead', 'DistanceToDriverBehind'])
print(pit_stop_laps['DistanceToDriverBehind'])

plt.figure(figsize=(10, 6))

plt.scatter(pit_stop_laps['DistanceToDriverAhead'], pit_stop_laps['PositionChange'], color='blue', label='Data points')

slope, intercept = np.polyfit(pit_stop_laps['DistanceToDriverAhead'], pit_stop_laps['PositionChange'], 1)
line = slope * pit_stop_laps['DistanceToDriverAhead'] + intercept

plt.plot(pit_stop_laps['DistanceToDriverAhead'], line, color='red', label='Fit line')

plt.title('Distance To Driver Ahead at time of pit stop vs. Position Change of overall race')
plt.xlabel('Distance To Driver Ahead on the lap the driver enters pits')
plt.ylabel('PositionChange')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))

# Scatter plot of mean DistanceToDriverAhead vs PositionChange
plt.scatter(pit_stop_df['DistanceToDriverBehind'], pit_stop_df['PositionChange'], color='blue', label='Data points')

plt.title('Distance To Driver Behind at time of pit stop vs. Position Change of overall race')
plt.xlabel('Distance To Driver Behind on the lap the driver enters pits')
plt.ylabel('PositionChange')
plt.legend()
plt.show()


