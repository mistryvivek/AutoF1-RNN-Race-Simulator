import datetime
import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import OneHotEncoder
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
    'Country': 'string', 'Location': 'string', 'OfficialEventName': 'string', 'EventName': 'string', 'Year': 'Int64',
    'DistanceToDriverAhead': 'Float64', 'DistanceToDriverBehind': 'Float64'
}

CSV_TO_FAST_F1_CONVERTERS = {
    **{col: lambda x: pd.to_timedelta(x, errors='coerce') if not pd.isna(x) else x 
       for col in FAST_F1_TIMEDELTA_COLUMNS},
    
    **{col: lambda x: x == "True" for col in FAST_F1_BOOLEAN_COLUMNS},
}

DNFS = ['R', 'D', 'E', 'W', 'F', 'N', 'U']

# Supress pd warnings.
pd.set_option('future.no_silent_downcasting', True)

class CustomF1Dataloader(Dataset):
    def generate_countdown_for_stint(self, group):  
        # Generate a sequence of integers from 1 to len(group)
        max_length = len(group)
        countdown = np.arange(1, max_length + 1)
        
        # Reverse the countdown sequence
        countdown = countdown[::-1]
        
        return pd.Series(countdown)

    def __init__(self, dataset_type, file_path):
        # Dataset type - 1, 2, 3 or 4
        # 1: all data ever
        # 2: all data apart from DNFs
        # 3: only data where points were scored
        # 4: only data where a driver gained a position
        # Traverse the directory
        self.lap_data = []
        self.largest_sequence_length = 0
        self.countdown_values_overall = []
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Loading {file_path}")

                    df = pd.read_csv(
                        file_path, 
                        index_col=0, 
                        dtype=CSV_COLUMN_DTYPES, 
                        parse_dates=FAST_F1_DATETIME_DTYPE,
                        date_format=lambda date_str: datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if date_str else pd.NaT,
                        converters=CSV_TO_FAST_F1_CONVERTERS
                    )
                    
                    COMPOUND_ENCODING = {"SOFT": 0.0, "MEDIUM": 1.0, "HARD": 2.0, "INTERMEDIATE": 3.0, "WET": 4.0, "UNKNOWN": -1.0}
                    # Encode compounds.
                    df['Compound'] = df['Compound'].fillna('UNKNOWN')
                    df['Compound'] = df['Compound'].replace({'HYPERSOFT': 'SOFT', 'SUPERSOFT': 'SOFT', 'ULTRASOFT': 'SOFT', 'TEST_UNKNOWN': 'UNKNOWN', 'TEST': 'UNKNOWN'})
                    # Do not want UNKNOWN to be encoded.
                    df['Compound'] = df['Compound'].apply(lambda tyre: COMPOUND_ENCODING[tyre])
                    
                    # Handle NA values for TyreLife.
                    df['ClassifiedPosition'] = df['ClassifiedPosition'].fillna('U') # for unknown
                    df['Points'] = df['Points'].fillna(0)
                    df['LapTime'] = df['LapTime'].dt.total_seconds()

                    # Some teamnames have been the changed over the past couples of years.
                    OLD_TEAMS_MAPPING = {'AlphaTauri': 'RB',
                                         'Toro Rosso': 'RB',
                                         'Renault': 'Alpine',
                                         'Force India': 'Aston Martin',
                                         'Racing Point': 'Aston Martin',
                                         'Alfa Romeo Racing': 'Kick Sauber',
                                         'Alfa Romeo': 'Kick Sauber',
                                         'Sauber': 'Kick Sauber'}
                    
                    TEAM_MAPPINGS = {'Red Bull Racing': 0, 'Alpine': 1, 'Aston Martin': 2, 'Ferrari': 3, 'Williams': 4, 'Haas F1 Team': 5, 'RB': 6, 'Kick Sauber': 7, 'McLaren': 8, 'Mercedes': 9, 'UNKNOWN': -1}
                    # HULK FP1
                    df['Team'] = df['Team'].fillna("UNKNOWN")
                    df['Team'] = df['Team'].apply(lambda team: TEAM_MAPPINGS[OLD_TEAMS_MAPPING[team]] if team in OLD_TEAMS_MAPPING.keys() else TEAM_MAPPINGS[team])
                   
                    df['TrackStatus'] = df['TrackStatus'].fillna("00000")
                    track_status_split = df['TrackStatus'].apply(lambda x: list(str(x).ljust(5, '0')))
                    track_status_df = pd.DataFrame(track_status_split.tolist(), columns=[f"TrackStatus{i+1}" for i in range(5)])
                    df = pd.concat([df, track_status_df], axis=1)
                    df = df.drop(columns=['TrackStatus'])

                    # Any drivers that have not raced are assigned the number 50.
                    # We could use driver number - but they sometimes change (e.g 1 given to last years winner.)
                    DRIVERS = ['VET', 'GAS', 'MSC', 'STR', 'ALB', 'FIT', 'SAR', 'GRO', 'ERI', 'PER', 'HAM', 'COL', 'LAW', 'AIT', 'KUB', 'LEC', 'PIA', 'RAI', 'OCO', 'LAT', 'GIO', 'NOR', 'BEA', 'SAI', 'HAR', 'MAG', 'TSU', 'ALO', 'VAN', 'BOT', 'RUS', 'DEV', 'MAZ', 'KVY', 'HUL', 'RIC', 'SIR', 'DOO', 'ZHO', 'VER']
                    df['Driver'] = df['Driver'].apply(lambda driver: DRIVERS.index(driver) if driver in DRIVERS else 50)

                    STATUS_MAPPINGS = {
                        # Finished Status (Success)
                        'Finished': 1, 'OnTrack': 1,

                        # Lap Behind Statuses
                        '+1 Lap': 2, '+2 Laps': 2, '+3 Laps': 2, '+6 Laps': 2, '+7 Laps': 2,

                        # Mechanical Issues
                        'Brakes': 3, 'Accident': 3, 'Collision damage': 3, 'Electrical': 3, 
                        'Wheel nut': 3, 'Driveshaft': 3, 'Turbo': 3, 'Gearbox': 3, 'Engine': 3, 
                        'Collision': 3, 'Vibrations': 3, 'Power Unit': 3, 'Hydraulics': 3, 'Suspension': 3, 
                        'Oil leak': 3, 'Rear wing': 3, 'Mechanical': 3, 'Power loss': 3, 'Puncture': 3, 
                        'Damage': 3, 'Overheating': 3, 'Undertray': 3, 'Steering': 3, 'Technical': 3, 
                        'Radiator': 3, 'Transmission': 3, 'Water pressure': 3, 'Fuel pressure': 3, 
                        'Water pump': 3, 'Cooling system': 3, 'Fuel leak': 3, 'Front wing': 3, 
                        'Water leak': 3, 'Fuel pump': 3, 'Differential': 3, 'Exhaust': 3, 'Battery': 3, 
                        'Tyre': 3, 'Electronics': 3, 'Debris': 3, 'Retired': 3, 'Wheel': 3,

                        # Driver/Team Error
                        'Spun off': 4, 'OffTrack': 4, 'Disqualified': 4, 'Out of fuel': 4, 

                        # Illness/Withdrawal
                        'Illness': 0, 'Withdrew': 0,

                        # Normally for non race events.
                        pd.NA: 5
                    }

                    df['Status'] = df['Status'].apply(lambda status: STATUS_MAPPINGS[status])

                    # Fill Qualifying times.
                    df['Q1'] = df['Q1'].fillna(0.0)
                    df['Q2'] = df['Q2'].fillna(0.0)
                    df['Q3'] = df['Q3'].fillna(0.0)

                    RACE_COLUMNS_TO_EXTRACT = [
                        "LapTime", "LapsTillPit", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "Speed",
                        "RPM", "nGear", "Throttle", "X", "Y", "Z",  # Telemetry
                        "DistanceToDriverAhead", "DistanceToDriverBehind", "TyreLife",
                        "AvgLapTimeP", "WorseLapTimeP", "LapTimeP", 
                        "AvgLapDiffP", "MandatoryPitStop", "Q1", "Q2", "Q3", # Other session data
                        "AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindDirection", "WindSpeed", # Weather
                        "MandatoryPitStop", "Status", "Compound", "Driver", "TrackStatus1", "TrackStatus2",
                        "TrackStatus3", "TrackStatus4", "TrackStatus5", "Team" # Anything that needs encoding
                    ]

                    for event in df['EventName'].unique():
                        dfEvent = df[df['EventName'] == event].copy()
                        # ========== FFILL DISTANCE DATA ===============
                        # Fill NaN with 0 if Position is 1 for 'DistanceToDriverAhead'
                        dfEvent['DistanceToDriverAhead'] = dfEvent.apply(
                            lambda row: 0 if pd.notna(row['Position']) and row['Position'] == 1 and pd.isna(row['DistanceToDriverAhead']) else row['DistanceToDriverAhead'], axis=1)

                        # Fill NaN with 0 if Position is 20 for 'DistanceToDriverBehind'
                        dfEvent['DistanceToDriverBehind'] = dfEvent.apply(
                            lambda row: 0 if pd.notna(row['Position']) and row['Position'] == dfEvent['Position'].max() and pd.isna(row['DistanceToDriverBehind']) else row['DistanceToDriverBehind'], axis=1)

                        for driver in dfEvent['Driver'].unique():
                            dfEventDriver = dfEvent[dfEvent['Driver'] == driver].copy()
                            dfEventDriverRace = dfEventDriver[dfEventDriver['Session'] == 'Race'].copy()
                            # ========== LOOK AT QUALIFYING DATA ========
                            if 'Sprint Shootout' in dfEventDriver['Session'].unique():
                                quali_times = dfEventDriver[dfEventDriver['Session'] == 'Sprint Shootout'].iloc[0][['Q1','Q2','Q3']].apply(lambda x: x if x == 0.0 else x.total_seconds()).values
                            else:
                                try:
                                    # Get Q1, Q2, Q3 times.
                                    quali_times = dfEventDriver[dfEventDriver['Session'] == 'Qualifying'].iloc[0][['Q1','Q2','Q3']].apply(lambda x: x if x == 0.0 else x.total_seconds()).values
                                except IndexError as e:
                                    quali_times = [0.0] * 3
                            # ========== LOOK AT PRACTICE DATA ========
                            dfPractice = dfEventDriver[dfEventDriver['Session'].str.contains('Practice', na=False)].copy().sort_values(by='Time')
                            avg_lap_time = dfPractice['LapTime'].mean()
                            worst_lap_time = dfPractice['LapTime'].max()
                            lap_time_sd = dfPractice['LapTime'].std()
                            # Compute Lap-to-Lap Differences
                            dfPractice.loc[:, 'LapDiff'] = dfPractice['LapTime'].diff() # Difference between consecutive laps
                            avg_lap_diff = dfPractice['LapDiff'].mean()
                            # Happens in the rare case there is no data here.
                            avg_lap_time = 0 if pd.isna(avg_lap_time) else avg_lap_time
                            worst_lap_time = 0 if pd.isna(worst_lap_time) else worst_lap_time
                            lap_time_sd = 0 if pd.isna(lap_time_sd) else lap_time_sd
                            avg_lap_diff = 0 if pd.isna(avg_lap_diff) else avg_lap_diff

                            if dfEventDriverRace.shape[0] > self.largest_sequence_length:
                                self.largest_sequence_length = dfEventDriverRace.shape[0]
                            if  (dfEventDriverRace.shape[0] > 1) and \
                                (not dfEventDriverRace['LapTime'].isna().any()) and \
                                (not ((dfEventDriverRace['Compound'] == -1.0).any())) and \
                                ((dataset_type == 1) or (dataset_type == 2 and dfEventDriverRace.iloc[0]['ClassifiedPosition'] not in DNFS) or \
                                (dataset_type == 3 and dfEventDriverRace.iloc[0]['Points'] > 0) or \
                                (dataset_type == 4 and dfEventDriverRace.iloc[0]['ClassifiedPosition'] not in DNFS and dfEventDriverRace.iloc[0]['GridPosition'] <= float(dfEventDriverRace.iloc[0]['ClassifiedPosition']))):
                                orderedLaps = dfEventDriverRace[(dfEventDriverRace['Driver'].astype(object) == driver)].sort_values(by='LapNumber')
                                # Apply the forward fill
                                # Can't fix warning - https://github.com/pandas-dev/pandas/issues/54417
                                orderedLaps['DistanceToDriverAhead'] = pd.to_numeric(orderedLaps['DistanceToDriverAhead'], errors='coerce').interpolate(method='nearest').fillna(df[df['Position'] == orderedLaps['Position'].iloc[0]]['DistanceToDriverAhead'].mean())
                                orderedLaps['DistanceToDriverBehind'] = pd.to_numeric(orderedLaps['DistanceToDriverBehind'], errors='coerce').interpolate(method='nearest').fillna(df[df['Position'] == orderedLaps['Position'].iloc[0]]['DistanceToDriverBehind'].mean())
                                orderedLaps['DistanceToDriverAhead'] = orderedLaps['DistanceToDriverAhead'] / 1000 # Convert to KM.
                                orderedLaps['DistanceToDriverBehind'] = orderedLaps['DistanceToDriverBehind'] / 1000

                                # ============ TRY STINT CHANGE =====================
                                # Initialize an empty list to store the countdown values
                                countdown_values = []

                                # Iterate through unique stints except the last stint
                                unique_stints = orderedLaps["Stint"].unique()

                                for stint in unique_stints[:-1]:  # Exclude the last stint
                                    # Get the group of rows that belong to this stint
                                    stint_group = orderedLaps[orderedLaps["Stint"] == stint]
                                    
                                    # Generate a sequence from 1 to the length of the stint group
                                    max_length = len(stint_group)
                                    countdown = np.arange(1, max_length + 1)  # Sequence from 1 to length of group
                                    
                                    # Reverse the countdown sequence (so it goes from max length to 1)
                                    countdown = countdown[::-1]
                                    
                                    # Add the reversed countdown to the list
                                    countdown_values.extend(countdown)
                                    self.countdown_values_overall.extend(countdown)

                                # Add the countdown for the last stint (set to 0)
                                last_stint = orderedLaps["Stint"].iloc[-1]
                                countdown_values.extend(np.arange(len(orderedLaps[orderedLaps["Stint"] == last_stint]), 0, -1))

                                # Set the countdown values for the DataFrame
                                orderedLaps["LapsTillPit"] = countdown_values

                                # ========= SORT OUT TELEMETRY DATA ==================
                                TELEMETRY_COLUMNS = ['Speed', 'RPM', 'nGear', 'Throttle', 'X', 'Y', 'Z']
                                for col in TELEMETRY_COLUMNS:
                                    orderedLaps[col] = orderedLaps[col].interpolate(method='nearest').fillna(df[df['Location'] == orderedLaps['Location'].iloc[0]][col].mean())

                                # ========= CONVERT RPM TO RPS ========================
                                orderedLaps['RPM'] = orderedLaps['RPM'] * (1/60)

                                # ============ SORT OUT SPEED DATA ===================
                                SPEED_COLUMNS = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
                                for col in SPEED_COLUMNS:
                                    orderedLaps[col] = orderedLaps[col].bfill().fillna(df[df['Location'] == orderedLaps['Location'].iloc[0]][col].mean())
                                
                                # ========== TELEMETRY (PERCENTAGE) ==================
                                orderedLaps["Throttle"] = orderedLaps["Throttle"] / 100

                                # ================= WEATHER FILL DATA ==================
                                # Forwardfill by using the last known point.
                                WEATHER_COLUMNS = ['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']
                                for col in WEATHER_COLUMNS:
                                    orderedLaps[col] = orderedLaps[col].bfill().fillna(df[df['Location'] == orderedLaps['Location'].iloc[0]][col].mean())

                                ## =========== CONVERT TO MINS FOR STABLITY =================
                                orderedLaps[["Q1", "Q2", "Q3"]] = quali_times
                                orderedLaps[["AvgLapTimeP", "WorseLapTimeP", "LapTimeP", "AvgLapDiffP"]] = [avg_lap_time, worst_lap_time, lap_time_sd, avg_lap_diff]
                                orderedLaps[["LapTime", "Q1", "Q2", "Q3", "AvgLapTimeP", "WorseLapTimeP", "LapTimeP", "AvgLapDiffP"]] = orderedLaps[["LapTime", "Q1", "Q2", "Q3", "AvgLapTimeP", "WorseLapTimeP", "LapTimeP", "AvgLapDiffP"]] / 60.0
                                self.lap_data.append(torch.tensor(orderedLaps[RACE_COLUMNS_TO_EXTRACT].to_numpy().astype('float32'), dtype=torch.float32))

    def __len__(self):
        return len(self.lap_data)

    def __getitem__(self, idx):
        """ Want to output:
        AUTOREGRESSIVE OUTPUTS
        1. Did we pit previous lap?
        2. Lap Time
        3. Compound
        5. Speed (x5)
        TELEMETRY
        6. Speed
        7. RPM
        8. nGear
        9. Throttle
        10. Brake
        11. DRS
        12. X 
        13. Y
        14. Z 
        15. Status
        -------
        DATA THAT NEED ENCODING:
        16. DriverNumber
        17. Team
        18. TrackStatus
        ------
        CONSTANT DATA
        WEATHER
        19.TyreLife
        20.AirTemp
        21.Humidity
        22.Pressure
        23.Rainfall
        24.TrackTemp
        25.WindDirection
        26.WindSpeed
        --------
        TIMING DATA
        27. Q1
        28. Q2
        29. Q3
        30. DistanceToDriverAhead
        31. DistanceToDriverBehind
        """
        return self.lap_data[idx]
