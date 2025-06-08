import requests
import csv
from datetime import datetime

# API Configuration
BASE_URL = "https://api.brightsky.dev/weather"
HEADERS = {"Accept": "application/json"}

def fetch_weather_data_from_api(start_date, end_date, dwd_station_id="15000"):
    """
    Fetches weather data from the Bright Sky API for a given date range and station.

    Args:
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
        dwd_station_id (str): The DWD station ID.

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    querystring = {
        "date": start_date,
        "last_date": end_date,
        "dwd_station_id": dwd_station_id,
        "units": "dwd"  # Requesting DWD units as per source table and query
    }
    try:
        print(f"Requesting data for station {dwd_station_id} from {start_date} to {end_date}...")
        response = requests.get(BASE_URL, headers=HEADERS, params=querystring, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        print("Data fetched successfully.")
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Error: API request timed out for station {dwd_station_id}.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error: API request failed with status {e.response.status_code} for station {dwd_station_id}.")
        print(f"Response: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: An issue occurred while fetching data from the API: {e}")
        return None

def transform_data_for_csv(api_data_item):
    """
    Transforms a single data item from the API response to the target CSV row format.
    Handles missing values and unit conversions.

    Args:
        api_data_item (dict): A dictionary representing one weather data point from the API.

    Returns:
        dict: A dictionary formatted for the CSV output.
    """
    
    # Helper to safely get values from the API item, providing a default if None
    def get_api_value(key, default_if_none=""):
        val = api_data_item.get(key)
        return val if val is not None else default_if_none

    # Timestamp formatting: API is ISO 8601 UTC -> dd/mm/yy HH:MM AM/PM
    formatted_timestamp = ""
    api_timestamp = api_data_item.get("timestamp")
    if api_timestamp:
        try:
            # Example: "2025-05-19T00:00:00+00:00"
            dt_object = datetime.fromisoformat(api_timestamp.replace("Z", "+00:00"))
            formatted_timestamp = dt_object.strftime('%d/%m/%y %I:%M %p')
        except ValueError:
            print(f"Warning: Could not parse timestamp: {api_timestamp}. Leaving it empty.")
            # Keep formatted_timestamp as ""

    # Solar irradiation conversion:
    # API 'solar' (DWD unit) is in kWh/m² (typically integrated over the hour).
    # Target CSV 'shortwave_radiation (W/m²)' is power.
    # Conversion: X kWh/m² over 1 hour = X * 1000 W/m² (average power).
    solar_kwh_m2 = api_data_item.get("solar")
    shortwave_radiation_w_m2 = 0  # Default to 0 if missing or not convertible (as in example CSV)
    if solar_kwh_m2 is not None:
        try:
            shortwave_radiation_w_m2 = float(solar_kwh_m2) * 1000
        except (ValueError, TypeError):
            # If conversion fails, it remains 0
            pass 
            
    csv_row = {
        "time": formatted_timestamp,
        "temperature_2m (°C)": get_api_value("temperature", "0.0"),
        "relative_humidity_2m (%)": get_api_value("relative_humidity", "0"),
        "dew_point_2m (°C)": get_api_value("dew_point", "0.0"),
        "apparent_temperature (°C)": "",  # Explicitly empty as per requirements
        "cloud_cover (%)": get_api_value("cloud_cover", "0"),
        "wind_speed_10m (km/h)": get_api_value("wind_speed", "0.0"), # DWD unit is km/h
        "wind_direction_10m (°)": get_api_value("wind_direction", "0"), # DWD unit is °
        "shortwave_radiation (W/m²)": shortwave_radiation_w_m2
    }
    return csv_row

def write_weather_data_to_csv(weather_records, output_filename):
    """
    Writes the transformed weather data to a CSV file.

    Args:
        weather_records (list): A list of dictionaries, where each dictionary is a row for the CSV.
        output_filename (str): The name of the CSV file to create.
    """
    if not weather_records:
        print("No data provided to write to CSV.")
        return

    # Define CSV headers based on the target format
    headers = [
        "time", "temperature_2m (°C)", "relative_humidity_2m (%)", 
        "dew_point_2m (°C)", "apparent_temperature (°C)", "cloud_cover (%)", 
        "wind_speed_10m (km/h)", "wind_direction_10m (°)", "shortwave_radiation (W/m²)"
    ]

    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(weather_records)
        print(f"✅ Weather data successfully written to {output_filename}")
    except IOError:
        print(f"❌ Error: Could not write to file {output_filename}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during CSV writing: {e}")

def process_weather_request_and_save_csv(start_date, end_date, output_csv_filename, dwd_station_id="15000"):
    """
    Main processing function: fetches data from API, transforms it, and writes to CSV.
    """
    api_response_json = fetch_weather_data_from_api(start_date, end_date, dwd_station_id)

    if api_response_json and "weather" in api_response_json:
        api_weather_data = api_response_json["weather"]
        
        if not api_weather_data:
            print("No weather data items found in the API response for the given parameters.")
            return

        transformed_records = []
        for item in api_weather_data:
            transformed_records.append(transform_data_for_csv(item))
        
        if transformed_records:
            write_weather_data_to_csv(transformed_records, output_csv_filename)
        else:
            print("No data was transformed, CSV file not created.")
    else:
        print("Failed to retrieve or parse valid weather data from the API. CSV file not created.")

# --- Main execution ---
if __name__ == "__main__":
    print("🌦️ Bright Sky Weather Data to CSV Generator 🌦️")
    print("-" * 40)

    # --- Configuration ---
    # You can modify these default values or the script will prompt you.
    start_date = "2025-05-19"
    end_date = "2025-05-29"
    dwd_station_id = "15000"  # As in your example request
    output_filename = f"./data/own_data/weather_{dwd_station_id}_{start_date}-{end_date}.csv"

    # run actual script
    process_weather_request_and_save_csv(
        start_date, 
        end_date, 
        output_filename,
        dwd_station_id
    )