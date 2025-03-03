import boto3
import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
import io

# AWS S3 bucket and folder paths
bucket_name = '298a'  # Replace with your S3 bucket name
input_folder = 'input/'  # Folder for input HTML files
output_folder = 'processed/'  # Folder for output CSV files

# YouTube API details
API_KEY = 'AIzaSyDy9Ffl9932AOdWnlWh8GZGle6ay6GeN8o'  # Replace with your actual API key
VIDEO_DETAILS_URL = 'https://www.googleapis.com/youtube/v3/videos'
CATEGORY_DETAILS_URL = 'https://www.googleapis.com/youtube/v3/videoCategories'

# Initialize S3 client
# s3 = boto3.client('s3')


# Function to parse the HTML file and extract video data
def extract_video_data(html_content):
    video_data = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all video entries (assuming each video has a link with a date-time in nearby text)
    for entry in soup.find_all('a', href=True):
        match = re.search(r'v=([a-zA-Z0-9_-]+)', entry['href'])
        if match:
            video_id = match.group(1)
            video_url = entry['href']
            date_time = entry.find_next_sibling(text=True)
            if date_time:
                date_time = date_time.strip()

            video_data.append({
                'video_id': video_id,
                'watched_at': date_time,
                'video_url': video_url
            })
    return video_data


# Function to fetch video details from the API
def get_video_details(video_id):
    try:
        params = {
            'part': 'snippet,contentDetails,statistics',
            'id': video_id,
            'key': API_KEY
        }
        response = requests.get(VIDEO_DETAILS_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to retrieve details for video ID {video_id}: {e}")
        return None


# Function to fetch category name from the API
def get_category_name(category_id):
    try:
        params = {
            'part': 'snippet',
            'id': category_id,
            'key': API_KEY
        }
        response = requests.get(CATEGORY_DETAILS_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data['items'][0]['snippet']['title'] if data['items'] else None
    except requests.RequestException as e:
        print(f"Failed to retrieve category name for category ID {category_id}: {e}")
        return None

def process_html(s3, bucket_name, file_key):
    # List all files in the input folder
    try:
        print(f"Processing file: {file_key}")
        
        # Download the HTML file from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        html_content = response['Body'].read().decode('utf-8')
        
        # with open(file_key, 'r', encoding='utf-8') as file:
        #     html_content = file.read()
        
    # Extract video watch data
        video_watch_data = extract_video_data(html_content)

        # List to store enriched video data
        final_video_data = []

        # Fetch details for each video
        for video in video_watch_data[:20]:
            video_id = video['video_id']
            video_url = video['video_url']
            watched_at = video['watched_at']

            video_info = get_video_details(video_id)
            if video_info is None or 'items' not in video_info or not video_info['items']:
                continue  # Skip if video details are unavailable

            video_item = video_info['items'][0]
            snippet = video_item.get('snippet', {})
            category_id = snippet.get('categoryId')
            category_name = get_category_name(category_id) if category_id else None

            # Append data to the final list
            final_video_data.append({
                'Video ID': video_id,
                'Video Link': video_url,
                'Title': snippet.get('title'),
                'Description': snippet.get('description'),
                'Category ID': category_id,
                'Category': category_name,
                'Watched At': watched_at
            })

        # Save the processed data to CSV
        df = pd.DataFrame(final_video_data)
        output_key = file_key.replace(input_folder, output_folder).replace('.html', '.csv')
        print(output_key, file_key)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        
        # Upload the CSV file back to S3
        s3.put_object(Bucket=bucket_name, Key=output_key, Body=buffer.getvalue())
        print(f"Processed file saved to: {output_key}")
    except Exception as e:
        print(f"Error processing file {file_key}: {e}")

    
# Lambda handler
# def lambda_handler(event, context):
    # List all files in the input folder
    # response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_folder)
    # files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.html')]
    
    # for file_key in files:
        # try:
        #     print(f"Processing file: {file_key}")
            
        #     # Download the HTML file from S3
        #     response = s3.get_object(Bucket=bucket_name, Key=file_key)
        #     html_content = response['Body'].read().decode('utf-8')
            
        #     # Extract video watch data
        #     video_watch_data = extract_video_data(html_content)

        #     # List to store enriched video data
        #     final_video_data = []

        #     # Fetch details for each video
        #     for video in video_watch_data[:20]:
        #         video_id = video['video_id']
        #         video_url = video['video_url']
        #         watched_at = video['watched_at']

        #         video_info = get_video_details(video_id)
        #         if video_info is None or 'items' not in video_info or not video_info['items']:
        #             continue  # Skip if video details are unavailable

        #         video_item = video_info['items'][0]
        #         snippet = video_item.get('snippet', {})
        #         category_id = snippet.get('categoryId')
        #         category_name = get_category_name(category_id) if category_id else None

        #         # Append data to the final list
        #         final_video_data.append({
        #             'Video ID': video_id,
        #             'Video Link': video_url,
        #             'Title': snippet.get('title'),
        #             'Description': snippet.get('description'),
        #             'Category ID': category_id,
        #             'Category': category_name,
        #             'Watched At': watched_at
        #         })

        #     # Save the processed data to CSV
        #     df = pd.DataFrame(final_video_data)
        #     output_key = file_key.replace(input_folder, output_folder).replace('.html', '.csv')
        #     buffer = io.StringIO()
        #     df.to_csv(buffer, index=False)
            
        #     # Upload the CSV file back to S3
        #     s3.put_object(Bucket=bucket_name, Key=output_key, Body=buffer.getvalue())
        #     print(f"Processed file saved to: {output_key}")
        # except Exception as e:
        #     print(f"Error processing file {file_key}: {e}")
 