import subprocess
import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from io import StringIO

OUTPUT_FILE = "gujarat_cotton_prices.csv"
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "downloads")
HTML_FILE = os.path.join(DOWNLOAD_DIR, "market_data.html")

def fetch_market_data():
    today = datetime.today()
    if today.day <= 7:
        # Last month
        month = (today.month - 1) if today.month > 1 else 12
        year = today.year if month != 12 else today.year - 1
    else:
        # Current month
        month = today.month
        year = today.year

    print(f"Fetching data for month={month}, year={year}")

    # Run Node.js Puppeteer script
    JS_SCRIPT = os.path.join(os.path.dirname(__file__), "market_fetch.js")
    subprocess.run(["node", JS_SCRIPT, str(month), str(year)], check=True)


    # Read and parse HTML
    if not os.path.exists(HTML_FILE):
        raise FileNotFoundError("HTML file not found after download.")

    with open(HTML_FILE, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError("No table found in HTML file.")

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    df_new = pd.DataFrame(rows, columns=headers)

    # Keep only needed columns
    keep_cols = [col for col in df_new.columns if "Market" in col or "Min Price" in col]
    df_new = df_new[keep_cols]

    # Merge with existing
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates()
    else:
        df_combined = df_new

    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Updated {OUTPUT_FILE} — total records: {len(df_combined)}")

if __name__ == "__main__":
    fetch_market_data()
