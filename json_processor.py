import json
from datetime import datetime
from collections import defaultdict, Counter
import csv
from datetime import timedelta
import pandas as pd
import io

# Predefined domain categories for common sites
DOMAIN_CATEGORIES = {
    "google.com": "Search Engine",
    "youtube.com": "Entertainment",
    "facebook.com": "Social Media",
    "twitter.com": "Social Media",
    "instagram.com": "Social Media",
    "linkedin.com": "Professional Networking",
    "github.com": "Development",
    "gmail.com": "Email",
    "outlook.com": "Email",
    "amazon.com": "Shopping",
    "netflix.com": "Entertainment",
    # Add more as needed
}

input_folder = 'input/'  # Folder for input HTML files
output_folder = 'processed/'  # Folder for output CSV files

# Function to get the category of a domain
def get_domain_category(domain):
    for key, category in DOMAIN_CATEGORIES.items():
        if key in domain:
            return category
    return "Other"  # Default category if domain not listed


# Convert microseconds timestamp to datetime
def convert_time(time_usec):
    return datetime.utcfromtimestamp(time_usec / 1_000_000)


# Load the JSON data
def load_history_data(s3, bucket_name, file_key):
    # Download the HTML file from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    json_content = response['Body'].read().decode('utf-8')
    data = json.loads(json_content)
    return data if isinstance(data, list) else data.get("Browser History", [])


def preprocess_data_with_details(data):
    daily_data = defaultdict(lambda: {
        "total_visits": 0,
        "distinct_pages": set(),
        "domains": Counter(),
        "categories": Counter(),
        "session_count": 0,
        "avg_session_duration": timedelta(0),
        "first_visit_url": None,
        "last_visit_url": None,
        "visit_times_by_hour": Counter(),
        "homepage_visits": 0,
        "deep_page_visits": 0,
        "new_domains_count": 0,
        "returning_domains_count": 0,
        "first_visit": None,
        "last_visit": None
    })

    visited_domains = set()

    for entry in data:
        visit_time = convert_time(entry["time_usec"])
        visit_date = visit_time.date()
        url = entry.get("url", "")
        domain = url.split('/')[2] if "://" in url else ""
        category = get_domain_category(domain)

        # Determine if it's a homepage or deeper page
        is_homepage = url.count('/') <= 3
        if is_homepage:
            daily_data[visit_date]["homepage_visits"] += 1
        else:
            daily_data[visit_date]["deep_page_visits"] += 1

        # Update new vs. returning domains
        if domain not in visited_domains:
            daily_data[visit_date]["new_domains_count"] += 1
            visited_domains.add(domain)
        else:
            daily_data[visit_date]["returning_domains_count"] += 1

        # Update other daily data
        daily_info = daily_data[visit_date]
        daily_info["total_visits"] += 1
        daily_info["distinct_pages"].add(url)
        daily_info["domains"][domain] += 1
        daily_info["categories"][category] += 1
        daily_info["visit_times_by_hour"][visit_time.hour] += 1

        # First and last visit URLs
        if daily_info["first_visit"] is None or visit_time < daily_info["first_visit"]:
            daily_info["first_visit"] = visit_time
            daily_info["first_visit_url"] = url
        if daily_info["last_visit"] is None or visit_time > daily_info["last_visit"]:
            daily_info["last_visit"] = visit_time
            daily_info["last_visit_url"] = url

    # Format session information and finalize data
    for date, summary in daily_data.items():
        summary["distinct_pages"] = len(summary["distinct_pages"])
        summary["avg_session_duration"] = summary["avg_session_duration"] / max(summary["session_count"], 1)
        summary["visit_times_by_hour"] = dict(summary["visit_times_by_hour"])

    return daily_data


# Save the enhanced daily summary data to a CSV
def save_summary_to_csv(s3, bucket_name, daily_data, output_key):
    # Use an in-memory buffer for the CSV file
    buffer = io.StringIO()
    
    # Define fieldnames for the CSV
    fieldnames = [
        "date", "total_visits", "distinct_pages",
        "top_domain_1", "top_domain_1_count",
        "top_domain_2", "top_domain_2_count",
        "top_domain_3", "top_domain_3_count",
        "top_category_1", "top_category_1_count",
        "top_category_2", "top_category_2_count",
        "top_category_3", "top_category_3_count",
        # "session_count", "avg_session_duration",
        # "visit_times_by_hour", 
        "homepage_visits",
        "deep_page_visits", "new_domains_count",
        "returning_domains_count",
        # "first_visit", 
        "first_visit_url",
        # "last_visit",
        "last_visit_url"
    ]
    
    # Write CSV data to the in-memory buffer
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    for date, summary in sorted(daily_data.items()):
        # Get top 3 most visited domains and categories
        top_domains = summary["domains"].most_common(3)
        top_categories = summary["categories"].most_common(3)

        top_domain_1, top_domain_1_count = (top_domains[0] if len(top_domains) > 0 else ("", 0))
        top_domain_2, top_domain_2_count = (top_domains[1] if len(top_domains) > 1 else ("", 0))
        top_domain_3, top_domain_3_count = (top_domains[2] if len(top_domains) > 2 else ("", 0))

        top_category_1, top_category_1_count = (top_categories[0] if len(top_categories) > 0 else ("", 0))
        top_category_2, top_category_2_count = (top_categories[1] if len(top_categories) > 1 else ("", 0))
        top_category_3, top_category_3_count = (top_categories[2] if len(top_categories) > 2 else ("", 0))

        # Format visit times by hour as a string for readability
        visit_times_by_hour_str = "; ".join([f"{hour}:00-{hour+1}:00 ({count})" for hour, count in sorted(summary["visit_times_by_hour"].items())])

        # Convert timedelta to a readable format (hh:mm:ss)
        avg_session_duration_str = str(summary["avg_session_duration"]) if isinstance(summary["avg_session_duration"], timedelta) else "00:00:00"

        writer.writerow({
            "date": date.strftime("%Y-%m-%d"),
            "total_visits": summary["total_visits"],
            "distinct_pages": summary["distinct_pages"],
            "top_domain_1": top_domain_1,
            "top_domain_1_count": top_domain_1_count,
            "top_domain_2": top_domain_2,
            "top_domain_2_count": top_domain_2_count,
            "top_domain_3": top_domain_3,
            "top_domain_3_count": top_domain_3_count,
            "top_category_1": top_category_1,
            "top_category_1_count": top_category_1_count,
            "top_category_2": top_category_2,
            "top_category_2_count": top_category_2_count,
            "top_category_3": top_category_3,
            "top_category_3_count": top_category_3_count,
            # "session_count": summary.get("session_count", 0),
            # "avg_session_duration": avg_session_duration_str,
            # "visit_times_by_hour": visit_times_by_hour_str,
            "homepage_visits": summary.get("homepage_visits", 0),
            "deep_page_visits": summary.get("deep_page_visits", 0),
            "new_domains_count": summary.get("new_domains_count", 0),
            "returning_domains_count": summary.get("returning_domains_count", 0),
            # "first_visit": summary["first_visit"].strftime("%H:%M:%S") if summary["first_visit"] else "",
            "first_visit_url": summary.get("first_visit_url", ""),
            # "last_visit": summary["last_visit"].strftime("%H:%M:%S") if summary["last_visit"] else "",
            "last_visit_url": summary.get("last_visit_url", "")
        })
    
    # Upload the CSV data to S3
    buffer.seek(0)  # Reset the buffer position to the start
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=buffer.getvalue())
    print(f"CSV file saved to S3: {output_key}")

# Main processing function
def process_json(s3, bucket_name, file_key):
    output_key = file_key.replace(input_folder, output_folder).replace('.json', '.csv')
    data = load_history_data(s3, bucket_name, file_key)
    daily_data = preprocess_data_with_details(data)
    save_summary_to_csv(s3, bucket_name, daily_data, output_key)
    # Save the processed data to CSV
    # df = pd.DataFrame(daily_data)
    # print(output_key, file_key)
    # buffer = io.StringIO()
    # df.to_csv(buffer, index=False)
    
    # Upload the CSV file back to S3
    # s3.put_object(Bucket=bucket_name, Key=output_key, Body=output_key)
    print(f"Processed file saved to: {output_key}")
