import datetime
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
    'nGear': 'Int64', 'Throttle': 'Float64', 'Brake': 'Float64', 'X': 'Float64', 'Y': 'Float64',
    'Z': 'Float64', 'Status': 'string', 'DriverAhead': 'string', 'DistanceToDriverAhead': 'Float64',
    'DistanceToDriverBehind': 'Float64', 'DriverBehind': 'string', 'AirTemp': 'Float64',
    'Humidity': 'Float64', 'Pressure': 'Float64', 'Rainfall': 'string', 'TrackTemp': 'Float64',
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
        self.labels = []
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
                    df['TyreLife'] = df['TyreLife'].fillna(0)
                    df['ClassifiedPosition'] = df['ClassifiedPosition'].fillna('U') # for unknown
                    df['Points'] = df['Points'].fillna(0)
                    df['LapTime'] = df['LapTime'].dt.total_seconds()
                    
                    for event in df['EventName'].unique():
                        dfEvent = df[df['EventName'] == event]
                        for driver in dfEvent['Driver'].unique():
                            dfEventDriver = dfEvent[dfEvent['Driver'] == driver]
                            dfEventDriverRace = dfEventDriver[dfEventDriver['Session'] == 'Race']
                            if  (not dfEventDriverRace.empty) and \
                                ((dataset_type == 1) or (dataset_type == 2 and dfEventDriverRace.iloc[0]['ClassifiedPosition'] not in DNFS) or \
                                (dataset_type == 3 and dfEventDriverRace.iloc[0]['Points'] > 0) or \
                                (dataset_type == 4 and dfEventDriverRace.iloc[0]['ClassifiedPosition'] not in DNFS and dfEventDriverRace.iloc[0]['GridPosition'] <= float(dfEventDriverRace.iloc[0]['ClassifiedPosition']))):
                                orderedLaps = dfEventDriverRace[(dfEventDriverRace['Driver'].astype(object) == driver)].sort_values(by='LapNumber')
                                orderedLaps['StintChange'] = orderedLaps['Compound'].shift(-1).where(orderedLaps['Stint'] != orderedLaps['Stint'].shift(-1), 0)
                                data_input_array = orderedLaps[data_fields].to_numpy().astype('float32')
                                data_output_array = orderedLaps[['LapTime', 'StintChange']].to_numpy().astype('float32')
                                self.lap_data.append(torch.tensor(data_input_array, dtype=torch.float32))
                                self.labels.append(torch.tensor(data_output_array, dtype=torch.float32))

    def __len__(self):
        return len(self.lap_data)

    def __getitem__(self, idx):
        return self.lap_data[idx], self.labels[idx]
    
custom_dataset = CustomF1Dataloader(1, "TyreLife,Compound", "../Data Gathering")
custom_dataset = CustomF1Dataloader(2, "TyreLife,Compound", "../Data Gathering")
custom_dataset = CustomF1Dataloader(3, "TyreLife,Compound", "../Data Gathering")
custom_dataset = CustomF1Dataloader(4, "TyreLife,Compound", "../Data Gathering")