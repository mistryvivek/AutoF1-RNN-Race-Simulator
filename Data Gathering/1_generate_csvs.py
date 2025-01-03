import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a given year.")

    # Add a single positional argument for the year
    parser.add_argument("year", type=int, help="Year to be processed")

    # Parse the arguments
    args = parser.parse_args()

    # Example output using the provided year
    print(f"The year you entered is: {args.year}")

if __name__ == "__main__":
    main()
