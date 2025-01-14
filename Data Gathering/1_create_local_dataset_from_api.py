import fastf1 as f1
f1.set_log_level('INFO')
import pandas as pd

# Year by year to prevent rate limit issue.
import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a given year.")

    # Add a single positional argument for the year
    parser.add_argument("year", type=int, help="Start year to be processed")

    # Parse the arguments
    args = parser.parse_args()
    year = args.year

    print(f"The years you entered is: {year}")

    event_dataframes = []
    #This is fixed - the biggest amount of sessions per weekend is 5.
    SESSION_COLUMNS = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

    # Ignored testing due to various factors - primarily to stop rogue data from sandbagging.
    # Helps us get details of all sessions inc. practice, sprint, qualifying and the race.
    event_schedule = f1.get_event_schedule(year, include_testing=False)
    
    CANCELLED_SESSIONS = [[2020, 2, 'Practice 3'],
                          [2020, 11, 'Practice 1'],
                          [2020, 11, 'Practice 2'],
                          [2019, 17, 'Practice 3'],
                          [2021, 15, 'Practice 3']]
    
    # For each events, get all the lap data here.
    for _, row in event_schedule.iterrows():
        # Load dataframes for all 5 sessions.
        for session_column in SESSION_COLUMNS:
            # Example of where we don't stick to this format: https://www.formula1.com/en/results/2020/races/1057/emilia-romagna/practice/0
            # https://github.com/theOehrly/Fast-F1/issues/672
            if row[session_column] == '' or [year, row['RoundNumber'], row[session_column]] in CANCELLED_SESSIONS: 
                continue

            session = f1.get_session(year, int(row['RoundNumber']), row[session_column])
            session.load()
            # https://docs.fastf1.dev/core.html#fastf1.core.Telemetry - Can merge weather data here as well!
            # Merge the telemetry data.
            combined_dataset = session.laps.reset_index(drop=True)
            results = session.results

            # Add columns we are expecting for telemetry data.
            TELEMETRY_DATA_COLUMNS = [
                'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS',  # Car data
                'X', 'Y', 'Z', 'Status',  # Position data
                'DriverAhead', 'DistanceToDriverAhead', # Compare position
                'DistanceToDriverBehind', 'DriverBehind' # Custom implementation columns
            ]

            combined_dataset[TELEMETRY_DATA_COLUMNS] = pd.NA

            for idx, lap in combined_dataset.iterrows():
                try:
                    # There are so many telemetry points - we are looking lap by lap so we just take the last one.
                    telemetry_data = lap.get_telemetry().add_driver_ahead()
                    telemetry_data = telemetry_data[TELEMETRY_DATA_COLUMNS]
                    combined_dataset.loc[idx, TELEMETRY_DATA_COLUMNS] = telemetry_data.iloc[-1]
                except:
                    pass

            weather_data = session.laps.get_weather_data().reset_index(drop=True)
            combined_dataset = pd.concat([combined_dataset, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1)

            combined_dataset['MandatoryPitStop'] = True
            # Mandatory pit stop made - is not ready here either.
            for driver in combined_dataset["Driver"].unique():
                    driversRace = combined_dataset[combined_dataset["Driver"] == driver].sort_values("LapNumber")

                    # Iterate through each lap for the current driver
                    for idx, lap in driversRace.iterrows():
                        if row[session_column] == "Race":
                            # Check if the lap's compound is WET or INTERMEDIATE
                            # OR if the compound is different from the first lap's compound
                            if lap["Compound"] == "WET" or lap["Compound"] == "INTERMEDIATE" or \
                            lap["Compound"] != driversRace.iloc[0]["Compound"]:
                                break  # Break if the condition is met
                            else:
                                # Update the MandatoryPitStop for the current lap if conditions are met
                                combined_dataset.loc[
                                    (combined_dataset["Driver"] == driver) & 
                                    (combined_dataset["LapNumber"] == lap["LapNumber"]), 
                                    "MandatoryPitStop"
                                ] = False

                        results = session.results
                        combined_dataset.loc[combined_dataset["Driver"] == driver, results.columns] = results[results['Abbreviation'] == driver].iloc[0].values

            # Only distance not built in is distance behind - but we can factor that in.     
            if row[session_column] == "Race" or row[session_column] == "Sprint":
                for idx, lap in combined_dataset.iterrows():
                    driver_ahead, distance_to_driver_ahead = lap["DriverAhead"], lap["DistanceToDriverAhead"]
                    combined_dataset.loc[
                            (combined_dataset["DriverNumber"] == driver_ahead) & 
                            (combined_dataset["LapNumber"] == lap["LapNumber"]), 
                            ["DriverBehind", "DistanceToDriverBehind"]
                        ] = [driver_ahead, distance_to_driver_ahead]
            
            combined_dataset['Session'] = row[session_column]
            # Fields to check what race and session it is.
            combined_dataset['Country'] = row['Country']
            combined_dataset['Location'] = row['Location']
            combined_dataset['OfficialEventName'] = row['OfficialEventName']
            combined_dataset['EventDate'] = row['EventDate']
            combined_dataset['EventName'] = row['EventName']            
            combined_dataset['Year'] = year

            # Ensure data is sorted by Driver and Timestamp
            combined_dataset = combined_dataset.sort_values(by=['Driver', 'Time'])
            combined_dataset[TELEMETRY_DATA_COLUMNS] = combined_dataset.groupby('Driver',group_keys=False)[TELEMETRY_DATA_COLUMNS].apply(lambda group: group.ffill())
            # Reset index if needed (apply can modify the index)
            combined_dataset = combined_dataset.reset_index(drop=True)
            event_dataframes.append(combined_dataset)


        if len(event_dataframes) > 0:
            combined_dataframes = pd.concat(event_dataframes)
            event_dataframes = [combined_dataframes]
            combined_dataframes.reset_index(drop=True).to_csv(f"dataset{year}.csv")

    combined_dataframes.to_csv("dataset.csv")
   
if __name__ == "__main__":
    main()