import datetime
import pandas as pd
import argparse
import os
import torch
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="Process a given year.")

# Add a single positional argument for the file path.
parser.add_argument("file_path", type=str, help="file path contains all csv that will be used in the dataloader.")
parser.add_argument("fields", type=str, help="fields to exclude within the csv you are looking at")

# Parse the arguments
args = parser.parse_args()
file_path = args.file_path
fields = args.fields.split(",")  
fields = [str(field) for field in fields]  

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

lap_data = []

# Traverse the directory
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
            df['Compound'] = df['Compound'].fillna('MISSING')           
            df['Compound'] = encoder.fit_transform(df['Compound']).astype(float)
            

            for event in df['EventName'].unique():
                df[]
                for driver in df['Driver'].unique():
                    data_array = df[(df['Driver'] == driver) & (df['EventName'] == event) & (df['Session'] == 'Race')].sort_values(by='LapNumber')[fields].to_numpy().astype('float32')
                    print(data_array)
                    tensor = torch.tensor(data_array, dtype=torch.float32)
                    print(tensor)