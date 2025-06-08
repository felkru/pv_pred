import pandas as pd

def process_pv_data(input_csv_path, output_csv_path):
    """
    Processes raw PV power data to merge all strings and resample to hourly average power output.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the processed output CSV file.
    """
    try:
        # Load the CSV data
        df = pd.read_csv(input_csv_path)

        # Convert 'last_changed' to datetime objects, assuming UTC as indicated by 'Z'
        df['last_changed'] = pd.to_datetime(df['last_changed'], errors='coerce', utc=True)

        # Drop rows where 'last_changed' couldn't be parsed
        df.dropna(subset=['last_changed'], inplace=True)

        # Convert 'state' to numeric. Coerce errors will turn non-numeric values into NaN.
        df['state'] = pd.to_numeric(df['state'], errors='coerce')
        df.dropna(subset=['state'], inplace=True) # Drop rows where 'state' is NaN after conversion

        # Set 'last_changed' as the DataFrame index for time-series operations
        df.set_index('last_changed', inplace=True)

        # Sort the index to ensure proper time-series operations
        df.sort_index(inplace=True)

        # Combine power readings from all entity_ids by summing them for each timestamp
        # This assumes that for any given timestamp, the 'state' values from different
        # entity_ids are additive and represent the total power at that exact moment.
        # If there are multiple entries for the same entity_id at the same timestamp,
        # pandas will sum them up. If you have duplicate timestamps for the same
        # entity_id and want to take an average or the first/last, you'd need
        # df.groupby(level=0)['state'].mean() or .first() etc. first.
        # For simplicity, assuming unique or aggregate-on-sum behavior for same-timestamp entries.
        total_power = df.groupby(df.index)['state'].sum()

        # Create a new DataFrame from the total_power Series with the original timestamp index
        total_power_df = pd.DataFrame({'total_power': total_power})

        # Reindex to a complete hourly range and fill missing values with the closest available reading.
        # We'll create a full hourly range from the min to max timestamp in the data.
        start_time = total_power_df.index.min().floor('h')
        end_time = total_power_df.index.max().ceil('h')
        full_hourly_range = pd.date_range(start=start_time, end=end_time, freq='h', tz='UTC')

        # Reindex the total_power_df to this full hourly range.
        # Values not present in total_power_df will become NaN.
        resampled_power = total_power_df.reindex(full_hourly_range)

        # Now, fill the NaNs. Using 'nearest' for filling, then taking the mean of potential
        # multiple 'nearest' values if an hour interval spans multiple original data points.
        # 'nearest' will look both backward and forward.
        filled_power = resampled_power['total_power'].interpolate(method='time', limit_direction='both')

        # If after interpolation there are still NaNs at the very beginning/end (e.g., no data before/after first/last point),
        # fill with ffill then bfill to ensure all are filled.
        filled_power = filled_power.ffill().bfill().fillna(0) # Final fallback to 0 if still any NaNs

        # The 'resample' operation itself, when using 'mean', is the most direct way to get average power.
        # Let's go back to using resample('h').mean() on the original data, but with a slight twist:
        # First, ensure that each entity_id's data is distinct before summing, then resample.
        # A simpler approach: sum all power states at each timestamp and then resample this combined power.
        # The key for power is .mean() during resampling, not .sum().
        # However, the previous approach of summing individual sensor readings at their original timestamps
        # and then taking the average of those sums for each hour is more robust if readings aren't exactly on the hour.

        # Let's revert to a slightly simpler and often more robust approach for power:
        # 1. Group by exact timestamp and sum all 'state' values (total instantaneous power).
        # 2. Resample this instantaneous total power to hourly frequency and take the mean.
        instantaneous_total_power = df.groupby(df.index)['state'].sum()

        # Now resample this series to hourly and take the mean of the power readings within each hour
        # This will give the average power for that hour.
        hourly_average_power = instantaneous_total_power.resample('h').mean()

        # Fill any hours that might have no data after resampling (e.g., if there were gaps)
        # We can use forward fill then backward fill to get the "closest available" value for power.
        hourly_average_power = hourly_average_power.ffill().bfill().fillna(0) # Fallback to 0 if no data at all

        # Ensure the index of hourly_average_power is explicitly a DatetimeIndex
        if not isinstance(hourly_average_power.index, pd.DatetimeIndex):
            hourly_average_power.index = pd.to_datetime(hourly_average_power.index, errors='coerce', utc=True)
            hourly_average_power.dropna(inplace=True)

        # Create a new DataFrame for the output format
        output_df = pd.DataFrame({
            'Date': hourly_average_power.index.strftime('%d/%m/%Y %H:%M:%S'),
            'Power Output (Watts)': hourly_average_power.astype(int) # Ensure power is integer and rename header
        })

        # Save the processed data to a new CSV file, including headers
        output_df.to_csv(output_csv_path, index=False) # Removed header=False

        print(f"Successfully processed data and saved to {output_csv_path}")
        print("\nHead of the processed data:")
        print(output_df.head())

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define your input and output file paths
    input_file = 'data/own_data/19_03_25-29_03_25-raw,unmerged.csv' 
    output_file = 'data/own_data/19_03_25-29_03_25.csv'

    process_pv_data(input_file, output_file)
    