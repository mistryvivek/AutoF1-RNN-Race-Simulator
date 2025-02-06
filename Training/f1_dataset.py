import datetime
import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

FAST_F1_BOOLEAN_COLUMNS = [
    'IsPersonalBest', 'FreshTyre', 'Deleted', 'FastF1Generated', 'IsAccurate', 
    'DRS', 'MandatoryPitStop'
]

FAST_F1_TIMEDELTA_COLUMNS = [
    'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
    'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 
    'Q1', 'Q2', 'Q3'
]

FAST_F1_DATETIME_DTYPE = [
    'Time', 'PitOutTime', 'PitInTime', 'LapStartTime', 
    'LapStartDate', 'EventDate'
]

CSV_COLUMN_DTYPES = {
    'Driver': 'string', 'DriverNumber': 'string', 'LapNumber': 'Int64', 'Stint': 'Int64',
    'SpeedI1': 'Float64', 'SpeedI2': 'Float64', 'SpeedFL': 'Float64', 'SpeedST': 'Float64',
    'Compound': 'string', 'TyreLife': 'Float64', 'Team': 'string', 'TrackStatus': 'string',
    'Position': 'Int64', 'DeletedReason': 'string', 'Speed': 'Float64', 'RPM': 'Float64',
    'nGear': 'Int64', 'Throttle': 'Float64', 'Brake': 'boolean', 'X': 'Float64', 'Y': 'Float64',
    'Z': 'Float64', 'Status': 'string', 'DriverAhead': 'string', 'DistanceToDriverAhead': 'Float64',
    'DistanceToDriverBehind': 'Float64', 'DriverBehind': 'string', 'AirTemp': 'Float64',
    'Humidity': 'Float64', 'Pressure': 'Float64', 'Rainfall': 'boolean', 'TrackTemp': 'Float64',
    'WindDirection': 'Float64', 'WindSpeed': 'Float64', 'BroadcastName': 'string', 'Abbreviation': 'string',
    'DriverId': 'string', 'TeamName': 'string', 'TeamColor': 'string', 'TeamId': 'string', 'FirstName': 'string',
    'LastName': 'string', 'FullName': 'string', 'HeadshotUrl': 'string', 'CountryCode': 'string',
    'ClassifiedPosition': 'string', 'GridPosition': 'Int64', 'Points': 'Float64', 'Session': 'string',
    'Country': 'string', 'Location': 'string', 'OfficialEventName': 'string', 'EventName': 'string', 'Year': 'Int64'
}

CSV_TO_FAST_F1_CONVERTERS = {
    **{col: lambda x: pd.to_timedelta(x, errors='coerce') if not pd.isna(x) else x 
       for col in FAST_F1_TIMEDELTA_COLUMNS},
    
    **{col: lambda x: x == "True" for col in FAST_F1_BOOLEAN_COLUMNS},
}

DNFS = ['R', 'D', 'E', 'W', 'F', 'N', 'U']

class CustomF1Dataloader(Dataset):
    def __init__(self, dataset_type, data_fields, file_path):
        # Dataset type - 1, 2, 3 or 4
        # 1: all data ever
        # 2: all data apart from DNFs
        # 3: only data where points were scored
        # 4: only data where a driver gained a position
        # Traverse the directory
        self.lap_data = []
        self.time_labels = []
        self.compound_labels = []
        self.largest_sequence_length = 0
        data_fields = data_fields.split(",")  
        data_fields = [str(field) for field in data_fields]
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)

                    df = pd.read_csv(
                        file_path, 
                        index_col=0, 
                        dtype=CSV_COLUMN_DTYPES, 
                        parse_dates=FAST_F1_DATETIME_DTYPE,
                        date_format=lambda date_str: datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if date_str else pd.NaT,
                        converters=CSV_TO_FAST_F1_CONVERTERS
                    )

                    encoder = LabelEncoder()
                    # Encode compounds.
                    df['Compound'] = df['Compound'].fillna('UNKNOWN')    
                    df['Compound'] = encoder.fit_transform(df['Compound']).astype(float)
                    
                    # Handle NA values for TyreLife.
                    df['ClassifiedPosition'] = df['ClassifiedPosition'].fillna('U') # for unknown
                    df['Points'] = df['Points'].fillna(0)
                    df['LapTime'] = df['LapTime'].dt.total_seconds()

                    # Set all speeds to 0 if not already set.
                    df['SpeedI1'] = df['SpeedI1'].fillna(0)
                    df['SpeedI2'] = df['SpeedI2'].fillna(0)
                    df['SpeedFL'] = df['SpeedFL'].fillna(0)
                    df['SpeedST'] = df['SpeedST'].fillna(0)

                    # Drop laps that aren't part of a stint
                    df = df.dropna(subset=['Stint'])

                    # Forwardfill by using the last known point.
                    weather_fill = ['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']
                    df[weather_fill] = df[weather_fill].ffill()

                    # Some teamnames have been the changed over the past couples of years.
                    OLD_TEAMS_MAPPING = {'AlphaTauri': 'RB',
                                         'Toro Rosso': 'RB',
                                         'Renault': 'Alpine',
                                         'Force India': 'Aston Martin',
                                         'Racing Point': 'Aston Martin',
                                         'Alfa Romeo': 'Kick Sauber',
                                         'Sauber': 'Kick Sauber'}
                    
                    df['Team'] = df['Team'].apply(lambda x: OLD_TEAMS_MAPPING[x] if x in OLD_TEAMS_MAPPING.keys() else x)
                    df['Team'] = df['Team'].fillna("UNKNOWN")
                    df['Team'] = encoder.fit_transform(df['Team']).astype(float)
                    df['TrackStatus'] = df['TrackStatus'].fillna("UNKNOWN")
                    df['TrackStatus'] = encoder.fit_transform(df['TrackStatus']).astype(float)

                    # Fill Qualifying times.
                    df['Q1'] = df['Q1'].fillna(0.0)
                    df['Q2'] = df['Q2'].fillna(0.0)
                    df['Q3'] = df['Q3'].fillna(0.0)

                    # Have a seperate unknown class for unknown telemetrics.
                    df['Brake'] = df['Brake'].map({True: 1, False: 0, None: -1})
                    TELEMETRY_COLUMNS = ['Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS', 'X', 'Y', 'Z', 'Status']
                    for col in TELEMETRY_COLUMNS:
                        df[col] = encoder.fit_transform(df[col].astype(str)) 

                    for event in df['EventName'].unique():
                        dfEvent = df[df['EventName'] == event]
                        for driver in dfEvent['Driver'].unique():
                            dfEventDriver = dfEvent[dfEvent['Driver'] == driver]
                            dfEventDriverRace = dfEventDriver[dfEventDriver['Session'] == 'Race']
                            try:
                                # Get Q1, Q2, Q3 times.
                                quali_times = dfEventDriver[dfEventDriver['Session'] == 'Qualifying'].iloc[0][['Q1','Q2','Q3']].apply(lambda x: x if x == 0.0 else x.total_seconds()).values
                            except IndexError as e:
                                quali_times = [0.0] * 3
                            if dfEventDriverRace.shape[0] > self.largest_sequence_length:
                                self.largest_sequence_length = dfEventDriverRace.shape[0]
                            print(dfEventDriverRace.shape[0])
                            if  (dfEventDriverRace.shape[0] > 1) and \
                                (not dfEventDriverRace['LapTime'].isna().any()) and \
                                ((dataset_type == 1) or (dataset_type == 2 and dfEventDriverRace.iloc[0]['ClassifiedPosition'] not in DNFS) or \
                                (dataset_type == 3 and dfEventDriverRace.iloc[0]['Points'] > 0) or \
                                (dataset_type == 4 and dfEventDriverRace.iloc[0]['ClassifiedPosition'] not in DNFS and dfEventDriverRace.iloc[0]['GridPosition'] <= float(dfEventDriverRace.iloc[0]['ClassifiedPosition']))):
                                orderedLaps = dfEventDriverRace[(dfEventDriverRace['Driver'].astype(object) == driver)].sort_values(by='LapNumber')
                                orderedLaps['StintChange'] = orderedLaps['Compound'].shift(-1).where(orderedLaps['Stint'] != orderedLaps['Stint'].shift(-1), 0)
                                orderedLaps = orderedLaps[:-1]
                                data_input_array = orderedLaps[data_fields].to_numpy().astype('float32')
                                repeated_quali_times = np.tile(quali_times, (data_input_array.shape[0], 1))
                                self.lap_data.append(torch.tensor(np.concatenate([data_input_array, repeated_quali_times], axis=1), dtype=torch.float32))
                                self.time_labels.append(torch.tensor(orderedLaps[['LapTime']].to_numpy().astype('float32'), dtype=torch.float32))
                                self.compound_labels.append(torch.tensor(orderedLaps[['StintChange']].to_numpy().astype('float32'), dtype=torch.long))

    def add_padding(self, seq):
        if seq.shape[0] != self.largest_sequence_length:
            padding = seq[-1].unsqueeze(0).repeat(self.largest_sequence_length - seq.shape[0], 1)
            padded_seq = torch.cat([seq, padding], dim=0)
            return padded_seq
        else:
            return seq
    
    def __len__(self):
        return len(self.lap_data)

    def __getitem__(self, idx):
        return self.add_padding(self.lap_data[idx]), self.add_padding(self.compound_labels[idx]), self.add_padding(self.time_labels[idx])
    
"""custom_dataset = CustomF1Dataloader(1, "TyreLife,Compound", "../Data Gathering")
custom_dataset = CustomF1Dataloader(2, "TyreLife,Compound", "../Data Gathering")
custom_dataset = CustomF1Dataloader(3, "TyreLife,Compound", "../Data Gathering")
custom_dataset = CustomF1Dataloader(4, "TyreLife,Compound", "../Data Gathering")"""