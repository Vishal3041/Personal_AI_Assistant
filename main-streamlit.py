import streamlit as st
import boto3
import os
import json
import sys

# Get AWS credentials from environment variables or Streamlit secrets
def get_aws_credentials():
    try:
        # First try to get from Streamlit secrets (for Streamlit Cloud)
        aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
        aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
        region_name = st.secrets["aws"]["region_name"]
    except:
        # Fallback to environment variables (for local development)
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = os.getenv("AWS_REGION", "us-east-2")

    if not all([aws_access_key_id, aws_secret_access_key]):
        st.error("AWS credentials not found. Please set them in environment variables or Streamlit secrets.")
        return None

    return {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "region_name": region_name
    }

# Initialize S3 client with credentials
credentials = get_aws_credentials()
if credentials:
    s3 = boto3.client(
        service_name='s3',
        **credentials
    )
else:
    st.error("Failed to initialize AWS client. Please check your credentials.")
    st.stop()

# lambda_client = boto3.client(
#     service_name='lambda',
#     region_name='us-east-2',  # Replace with your Lambda region
#     aws_access_key_id='AKIAYS2NWIABPANVN3H5',
#     aws_secret_access_key='2IvszGuqMOQbL4FAp/Slek4nNkiwiXRj0OEd/eO0'
# )
def main(s3_key, bucket_name="298a"):
    # Get the file details from the event
    file_name = os.path.basename(s3_key)
    file_extension = os.path.splitext(file_name)[1].lower()

    # Process the file based on its extension
    if file_extension == '.html':
        from html_processor import process_html
        result = process_html(s3, bucket_name, s3_key)
    elif file_extension == '.json':
        from json_processor import process_json
        result = process_json(s3, bucket_name, s3_key)
    elif file_extension == '.csv':
        from csv_processor import process_csv
        result = process_csv(s3, bucket_name, s3_key)
    elif file_extension == '.ics':
        from ics_processor import process_ics
        result = process_ics(s3, bucket_name, s3_key, "abcd", "example@gmail.com")
    else:
        result = f"Unsupported file type: {file_extension}"

    return {
        "statusCode": 200,
        "body": result
    }

# Streamlit UI
st.title("File Processing Pipeline")

# File uploader (only shows if file type is selected)
uploaded_files = st.file_uploader(f"Upload your file", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
    # Display file details
        st.write("File {uploaded_file.name} uploaded successfully!")
        
        # Save the uploaded file locally
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Upload to S3
        s3_key = f"input/{uploaded_file.name}"  # Path in S3 bucket
        s3.upload_file(file_path, "298a", s3_key)
        st.write(f"File uploaded to S3 bucket: 298a, key: {s3_key}")
        
        # main(s3_key)

        response = main(s3_key)
        st.write(f"Processing Result: {response['body']}")
        
        try:
            os.remove(file_path)
            st.write("Temporary file deleted successfully.")
        except OSError as e:
            st.error(f"Error deleting temporary file: {e}")
    
    st.write("All files have been processed successfully.")


sys.exit()
