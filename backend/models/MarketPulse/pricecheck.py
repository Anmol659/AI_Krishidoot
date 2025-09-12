from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time


def scrape_krishi_dunia(commodity, state="Punjab"):
    # Convert rice -> paddy
    if commodity.lower() == "rice":
        commodity = "paddy"

    url = f"https://www.krishidunia.com/en/mandirates/{commodity}/{state}"
    print(f"Fetching: {url}")

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    time.sleep(5)  # allow JS to load
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    table = soup.find("table")
    if not table:
        raise ValueError("No table found")

    # Headers
    headers = [th.get_text(strip=True) for th in table.find("tr").find_all("th")]

    # Rows
    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)

    # Clean column names for backend
    df.rename(columns={
        "Crop": "crop",
        "Variety": "variety",
        "District": "district",
        "Market": "market",
        "Min PricePer/Quintal": "min_price",
        "Max PricePer/Quintal": "max_price",
        "Update Date": "update_date"
    }, inplace=True)

    # Clean price fields (remove ₹, commas)
    df["min_price"] = df["min_price"].str.replace("₹", "").str.replace(",", "").astype(float)
    df["max_price"] = df["max_price"].str.replace("₹", "").str.replace(",", "").astype(float)

    # Add metadata
    df["commodity"] = commodity.capitalize()
    df["state"] = state.capitalize()
    df["scraped_on"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df


if __name__ == "__main__":
    paddy_df = scrape_krishi_dunia("paddy", "Punjab")
    print(paddy_df.head())
