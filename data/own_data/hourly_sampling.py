import pandas as pd
from io import StringIO

def convert_pv_data_from_df(input_df):
    """
    Converts a DataFrame of PV power data into a structured hourly output
    by averaging the power readings within each hour.

    Args:
        input_df (pd.DataFrame): A DataFrame containing the raw PV data
                                  with 'entity_id', 'state', and 'last_changed' columns.

    Returns:
        pd.DataFrame: A DataFrame with 'Date' and 'Power Output (Watts)' columns.
    """
    df = input_df.copy()

    # Convert 'last_changed' to datetime objects, assuming UTC
    df['last_changed'] = pd.to_datetime(df['last_changed'], errors='coerce', utc=True)
    df.dropna(subset=['last_changed'], inplace=True)

    # Convert 'state' to numeric. Coerce errors will turn non-numeric values into NaN.
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df.dropna(subset=['state'], inplace=True) # Drop rows where 'state' is NaN after conversion

    # Set 'last_changed' as the DataFrame index for time-series operations
    df.set_index('last_changed', inplace=True)
    df.sort_index(inplace=True)

    # Calculate instantaneous total power by summing states at each timestamp.
    # This step is crucial if you have multiple sensors contributing to the total power
    # at the same exact timestamp.
    instantaneous_total_power = df.groupby(df.index)['state'].sum()

    # Resample to hourly frequency and take the mean of the power readings within each hour.
    # This directly calculates the average power for each 60-minute interval.
    hourly_average_power = instantaneous_total_power.resample('h').mean()

    # Fill any hours that might have no data after resampling (e.g., if there were large gaps).
    # We'll use forward fill then backward fill to propagate the last known good value,
    # and finally fill any remaining NaNs (e.g., at the very start) with 0.
    hourly_average_power = hourly_average_power.ffill().bfill().fillna(0)

    # Create the output DataFrame
    output_df = pd.DataFrame({
        'Date': hourly_average_power.index.strftime('%d/%m/%Y %H:%M:%S'),
        'Power Output (Watts)': hourly_average_power.astype(int)
    })

    return output_df

if __name__ == "__main__":
    # Define your input and output CSV file paths
    input_csv_path = 'data/own_data/19_05_25-29_05_25-raw.csv' # Make sure this CSV file exists in the same directory
    output_csv_path = 'data/own_data/19_05_25-29_05_25.csv'

    try:
        # Read the input CSV into a DataFrame
        raw_df = pd.read_csv(input_csv_path)

        # Process the DataFrame using the function
        converted_df = convert_pv_data_from_df(raw_df)

        # Write the processed DataFrame to a new CSV file
        converted_df.to_csv(output_csv_path, index=False)

        print(f"Successfully converted data from '{input_csv_path}' to '{output_csv_path}'.")
        print("\nHead of the converted data:")
        print(converted_df.head())

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'.")
        print("Please ensure 'input_pv_data.csv' is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")