import fastf1 as f1
f1.set_log_level('INFO')
import pandas as pd

# Year by year to prevent rate limit issue.
import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a given year.")

    # Add a single positional argument for the year
    parser.add_argument("year", type=int, help="Year to be processed")

    # Parse the arguments
    args = parser.parse_args()
    year = args.year

    # Example output using the provided year
    print(f"The year you entered is: {args.year}")
    
    # Ignored testing due to various factors - primarily to stop rogue data from sandbagging.
    # Helps us get details of all sessions inc. practice, sprint, qualifying and the race.
    event_schedule = f1.get_event_schedule(year, include_testing=False)
    #This is fixed - the biggest amount of sessions per weekend is 5.
    SESSION_COLUMNS = ['Session5']
    master_data = None
    
    # For each events, get all the lap data here.
    for _, row in event_schedule.iterrows():
        # Load dataframes for all 5 sessions.
        for session_column in SESSION_COLUMNS:
            # Example of where we don't stick to this format: https://www.formula1.com/en/results/2020/races/1057/emilia-romagna/practice/0
            if row[session_column] == '':
                continue
            session = f1.get_session(year, int(row['RoundNumber']), row[session_column])
            session.load()
            # https://docs.fastf1.dev/core.html#fastf1.core.Telemetry - Can merge weather data here as well!
            # Merge the telemetry data.
            combined_dataset = session.laps.reset_index(drop=True)
            
            # Add columns we are expecting for telemetry data.
            telemetry_data_columns = [
                'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS',  # Car data
                'X', 'Y', 'Z', 'Status',  # Position data
                'DriverAhead', 'DistanceToDriverAhead' # Compare position
            ]

            combined_dataset[telemetry_data_columns] = pd.NA

            for idx, lap in combined_dataset.iterrows():
                # There are so many telemetry points - we are looking lap by lap so we just take the last one.
                print(lap)
                telemetry_data = lap.get_telemetry()
                telemetry_data = telemetry_data[telemetry_data_columns]
                combined_dataset.loc[idx, telemetry_data_columns] = telemetry_data.iloc[-1]

            weather_data = session.laps.get_weather_data().reset_index(drop=True)
            combined_dataset = pd.concat([combined_dataset, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1)

            combined_dataset['MandatoryPitStop'] = True
            # Mandatory pit stop made - is not ready here either.
            if row[session_column] == "Race":
                for driver in combined_dataset["Driver"].unique():
                    driversRace = combined_dataset[combined_dataset["Driver"] == driver].sort_values("LapNumber")
                    
                    # Iterate through each lap for the current driver
                    for idx, lap in driversRace.iterrows():
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

                    """
                    results = session.results
                    for col in results.columns:
                        if pd.api.types.is_timedelta64_dtype(combined_dataset[col]):
                            combined_dataset.loc[combined_dataset["Driver"] == driver, col] = results[results['Abbreviation'] == driver][col].fillna(pd.Timedelta(0))
                        else:
                            combined_dataset.loc[combined_dataset["Driver"] == driver, col] = results[results['Abbreviation'] == driver][col]
                    """

                    combined_dataset.to_csv("test.csv")

            # Only distance not built in is distance behind - but we can factor that in.     
            if row[session_column] == "Race" or row[session_column] == "Sprint":
                for idx, lap in combined_dataset.iterrows():
                    driver_ahead, distance_to_driver_ahead = lap["DriverAhead"], lap["DistanceToDriverAhead"]
                    combined_dataset.loc[
                            (combined_dataset["DriverNumber"] == driver_ahead) & 
                            (combined_dataset["LapNumber"] == lap["LapNumber"]), 
                            ["DriverBehind", "DistanceToDriverBehind"]
                        ] = [driver_ahead, distance_to_driver_ahead]

            # Fields to check what race and session it is.
            combined_dataset['Country'] = row['Country']
            combined_dataset['Location'] = row['Location']
            combined_dataset['OfficialEventName'] = row['OfficialEventName']
            combined_dataset['EventDate'] = row['EventDate']
            combined_dataset['EventName'] = row['EventName']
            combined_dataset['Session'] = row[session_column]
            combined_dataset['Year'] = year

            combined_dataset.to_csv("test.csv")

            exit()
   
if __name__ == "__main__":
    main()