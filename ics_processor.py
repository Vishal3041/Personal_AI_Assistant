import csv
from icalendar import Calendar
import pandas as pd
from datetime import datetime
import os
import io


input_folder = 'input/'  # Folder for input HTML files
output_folder = 'processed/'  # Folder for output CSV files

# Parse the ICS file and append events to a DataFrame
def parse_ics_to_df(s3, bucket_name, file_key, name, email):
    events = []

    try:
        # Download the .ics file from S3 into memory
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        ics_content = response['Body'].read().decode('utf-8')  # Read and decode the file content
        
        # Parse the .ics file using icalendar
        calendar = Calendar.from_ical(ics_content)

        # Extract events from the calendar
        for component in calendar.walk():
            if component.name == "VEVENT":
                try:
                    event_name = component.get('summary', 'No Title')

                    # Handle Start Date and End Date with timezone conversion
                    start_date = component.get('dtstart').dt if component.get('dtstart') else ''
                    end_date = component.get('dtend').dt if component.get('dtend') else ''

                    # Convert timezone-aware dates to timezone-naive
                    if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                        start_date = start_date.replace(tzinfo=None)
                    if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                        end_date = end_date.replace(tzinfo=None)

                    description = component.get('description', 'No Description')
                    location = component.get('location', 'No Location')

                    # Store event in list
                    events.append({
                        'Name': name,
                        'Email': email,
                        'Event Name': event_name,
                        'Start Date': start_date,
                        'End Date': end_date,
                        'Description': description,
                        'Location': location
                    })
                except Exception as e:
                    print(f"Error processing event: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of errors

    # Convert events list to DataFrame
    return pd.DataFrame(events)

def save_df_to_s3(s3, df, bucket_name, output_key):
    try:
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
        df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')

        # Save DataFrame to an in-memory buffer as CSV
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)  # Reset buffer position

        # Upload the CSV to S3
        s3.put_object(Bucket=bucket_name, Key=output_key, Body=buffer.getvalue())
        print(f"CSV file successfully saved to s3://{bucket_name}/{output_key}")
    except Exception as e:
        print(f"Error saving CSV to S3: {e}")


def process_ics(s3, bucket_name, file_key, name, email):
    # output_csv_file = 'merged_calendar_events.csv'
    output_key = file_key.replace(input_folder, output_folder).replace('.ics', '.csv')
    # response = s3.get_object(Bucket=bucket_name, Key=file_key)
    # Step 1: Preprocess calendar data (Parse .ics files)
    preprocessed_data = parse_ics_to_df(s3, bucket_name, file_key, name, email)
    
    if not preprocessed_data.empty:
        # Save the resulting DataFrame as a CSV file to S3
        save_df_to_s3(s3, preprocessed_data, bucket_name, output_key)
    else:
        print("No events found or an error occurred.")