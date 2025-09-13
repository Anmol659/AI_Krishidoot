import requests
import sys

# ✅ Ensure proper UTF-8 encoding for output
sys.stdout.reconfigure(encoding="utf-8")

# Backend API endpoint
url = "https://backend-sewa.onrender.com/price"

state = "punjab"
commodities = ["wheat", "rice"]

for commodity in commodities:
    print(f"\n🔹 Fetching prices for {commodity.title()} in {state.title()}...\n")
    response = requests.get(url, params={"state": state, "commodity": commodity})

    if response.status_code == 200:
        data = response.json()
        for row in data["prices"]:
            print(row)   # ✅ Rupee symbol will now print correctly
    else:
        print(f" Error {response.status_code}: {response.text}")
