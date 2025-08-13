# -*- coding: utf-8 -*-
import sys
import requests

# Force UTF-8 encoding for stdout on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Your API key and Resource ID
API_KEY = "579b464db66ec23bdd000001a24d5db1ff3a4c906c0a7defd78c3262"
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# API URL with filters
url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
params = {
    "api-key": API_KEY,
    "format": "json",
    "limit": 10,
    "filters[commodity]": "Cotton",
    "filters[state]": "Gujarat"
}

# Request data
response = requests.get(url, params=params)
data = response.json()

# Print output
if "records" in data and len(data["records"]) > 0:
    print("Live Cotton Prices in Gujarat (Agmarknet):\n")
    for record in data["records"]:
        print(f"Date: {record.get('arrival_date')}")
        print(f"District: {record.get('district')}")
        print(f"Market: {record.get('market')}")
        print(f"Modal Price: ₹{record.get('modal_price')} per quintal")
        print(f"Min Price: ₹{record.get('min_price')} | Max Price: ₹{record.get('max_price')}")
        print("-" * 50)
else:
    print(" No data found for Cotton in Gujarat.")
