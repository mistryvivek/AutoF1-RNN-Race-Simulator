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
    event_schedule.to_csv("test.csv")

    #This is fixed - the biggest amount of sessions per weekend is 5.
    SESSION_COLUMNS = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    
    # For each events, get all the lap data here.
    for _, row in event_schedule.iterrows():
        # Load dataframes for all 5 sessions.
        for session_column in SESSION_COLUMNS:
            # Example of where we don't stick to this format: https://www.formula1.com/en/results/2020/races/1057/emilia-romagna/practice/0
            if row[session_column] is '':
                continue
            session = f1.get_session(year, int(row['RoundNumber']), row[session_column])
            session.load()
            # https://docs.fastf1.dev/core.html#fastf1.core.Telemetry - Can merge weather data here as well!
            # Merge the telemetry data.            

if __name__ == "__main__":
    main()
