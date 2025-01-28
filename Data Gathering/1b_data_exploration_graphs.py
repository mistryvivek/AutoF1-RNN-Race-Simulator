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
print(combined_dataset_corr_cols['LapTime'].apply(type).value_counts())
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




