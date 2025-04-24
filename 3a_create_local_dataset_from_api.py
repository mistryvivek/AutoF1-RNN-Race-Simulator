"""
Command line script to create F1 Dataset in a CSV for year entered as input.

python3 1_create_local_dataset_from_api.py <year>
"""

import fastf1 as f1
f1.set_log_level('INFO')
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process a given year.")
    parser.add_argument("year", type=int, help="Year to be processed.")
    args = parser.parse_args()
    year = args.year

    print(f"The year you entered is: {year}")

    # Storage for the collected for data for the year.
    event_dataframes = []
    # According to the library documentation, these are the only session
    # heading at the time of coding.
    SESSION_COLUMNS = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

    # Ignored testing due to various factors - primarily to stop rogue data from sandbagging.
    # Helps us get details of all sessions inc. practice, sprint, qualifying and the race.
    event_schedule = f1.get_event_schedule(year, include_testing=False)

    # If we try load these session, it will through an error as they were cancelled.
    CANCELLED_SESSIONS = [[2020, 2, 'Practice 3'],
                          [2020, 11, 'Practice 1'],
                          [2020, 11, 'Practice 2'],
                          [2019, 17, 'Practice 3'],
                          [2021, 15, 'Practice 3']]

    for _, row in event_schedule.iterrows():
        for session_column in SESSION_COLUMNS:
            if row[session_column] == '' or [year, row['RoundNumber'], row[session_column]] in CANCELLED_SESSIONS: 
                continue

            session = f1.get_session(year, int(row['RoundNumber']), row[session_column])
            session.load()
            combined_dataset = session.laps.reset_index(drop=True)
            results = session.results

            GIVEN_TELEMETRY_DATA_COLUMNS = [
                'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS',  # Car data
                'X', 'Y', 'Z', 'Status',  # Position data
                'DriverAhead', 'DistanceToDriverAhead' # Compare position
            ]

            # Need to create this columns for all rows before we add the telemetry data one by one.
            combined_dataset[GIVEN_TELEMETRY_DATA_COLUMNS] = pd.NA

            for idx, lap in combined_dataset.iterrows():
                try:
                    # There are so many telemetry points - we are looking lap by lap so we just take the last one.
                    telemetry_data = lap.get_telemetry().add_driver_ahead()
                    telemetry_data = telemetry_data[GIVEN_TELEMETRY_DATA_COLUMNS]
                    combined_dataset.loc[idx, GIVEN_TELEMETRY_DATA_COLUMNS] = telemetry_data.iloc[-1][GIVEN_TELEMETRY_DATA_COLUMNS].values
                except:
                    pass

            weather_data = session.laps.get_weather_data().reset_index(drop=True)
            combined_dataset = pd.concat([combined_dataset, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1)

            combined_dataset['MandatoryPitStop'] = True
            # Create 'MandatoryPitStopMade' as described in paper.
            for driver in combined_dataset["Driver"].unique():
                    driversRace = combined_dataset[combined_dataset["Driver"] == driver].sort_values("LapNumber")

                    for idx, lap in driversRace.iterrows():
                        if row[session_column] == "Race":
                            # Check if the lap's compound is WET or INTERMEDIATE
                            # OR if the compound is different from the first lap's compound
                            if lap["Compound"] == "WET" or lap["Compound"] == "INTERMEDIATE" or \
                            lap["Compound"] != driversRace.iloc[0]["Compound"]:
                                break
                            else:
                                combined_dataset.loc[
                                    (combined_dataset["Driver"] == driver) & 
                                    (combined_dataset["LapNumber"] == lap["LapNumber"]), 
                                    "MandatoryPitStop"
                                ] = False

                        results = session.results
                        combined_dataset.loc[combined_dataset["Driver"] == driver, results.columns] = results[results['Abbreviation'] == driver].iloc[0].values

            # Add distance to driver behind - as detailed in the report.
            if row[session_column] == "Race" or row[session_column] == "Sprint":
                for idx, lap in combined_dataset.iterrows():
                    driver_ahead, distance_to_driver_ahead = lap["DriverAhead"], lap["DistanceToDriverAhead"]
                    combined_dataset.loc[
                            (combined_dataset["DriverNumber"] == driver_ahead) & 
                            (combined_dataset["LapNumber"] == lap["LapNumber"]), 
                            ["DriverBehind", "DistanceToDriverBehind"]
                        ] = [driver_ahead, distance_to_driver_ahead]
            
            combined_dataset['Session'] = row[session_column]
            combined_dataset['Country'] = row['Country']
            combined_dataset['Location'] = row['Location']
            combined_dataset['OfficialEventName'] = row['OfficialEventName']
            combined_dataset['EventDate'] = row['EventDate']
            combined_dataset['EventName'] = row['EventName']            
            combined_dataset['Year'] = year

            combined_dataset = combined_dataset.reset_index(drop=True)
            event_dataframes.append(combined_dataset)


        if len(event_dataframes) > 0:
            combined_dataframes = pd.concat(event_dataframes)
            event_dataframes = [combined_dataframes]
            combined_dataframes.reset_index(drop=True).to_csv(f"dataset{year}.csv")
   
if __name__ == "__main__":
    main()