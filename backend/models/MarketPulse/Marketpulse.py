import time
import pandas as pd
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from bs4 import BeautifulSoup

# --- Configuration ---
# Ensure chromedriver.exe is in the same folder as this script
CHROME_DRIVER_PATH = r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/MarketPulse/chromedriver.exe"
AGMARKNET_URL = "https://agmarknet.gov.in/SearchCmmMkt.aspx"

def get_latest_market_data(commodity="Cotton", state="Gujarat", search_days=14) -> pd.DataFrame:
    """
    Finds the most recent day with available data on Agmarknet by checking
    backwards from today, ensuring authenticity and providing the latest data.

    Args:
        commodity (str): The crop to search for.
        state (str): The state to search in.
        search_days (int): The maximum number of past days to check for data.

    Returns:
        A pandas DataFrame with the latest available price data.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    service = Service(executable_path=CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)

    # Configure stealth mode to appear as a human user
    stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32")

    # Loop backwards from today to find the most recent data
    for i in range(search_days):
        target_date = date.today() - timedelta(days=i)
        date_str = target_date.strftime("%d/%m/%Y")
        
        try:
            print(f"\n--- Checking for data on: {date_str} ---")
            driver.get(AGMARKNET_URL)
            wait = WebDriverWait(driver, 20) # Wait up to 20 seconds

            # Use JavaScript to reliably set the date
            date_input = wait.until(EC.element_to_be_clickable((By.ID, "txtDate")))
            driver.execute_script(f"arguments[0].value = '{date_str}';", date_input)
            
            # Select commodity and state
            Select(driver.find_element(By.ID, "ddlCommodity")).select_by_visible_text(commodity)
            Select(driver.find_element(By.ID, "ddlState")).select_by_visible_text(state)
            
            # Click the 'Go' button to submit the form
            driver.find_element(By.ID, "btnGo").click()

            # Wait for the results table to be present in the HTML
            wait.until(EC.presence_of_element_located((By.ID, 'cphBody_GridArrivalData')))
            
            # Now that the table exists, check if it actually contains data rows
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            table = soup.find('table', id='cphBody_GridArrivalData')
            
            # A table with data will have more than 2 rows (header rows + data rows)
            if table and len(table.find_all('tr')) > 2:
                print(f"Success! Data found for {date_str}. Parsing data...")
                df = pd.read_html(str(table))[0]
                
                # Clean the DataFrame
                df = df.iloc[1:, :]
                df.columns = ["Sl no.", "District Name", "Market Name", "Variety", "Group", 
                              "Arrivals (Tonnes)", "Min Price (Rs/Quintal)", 
                              "Max Price (Rs/Quintal)", "Modal Price (Rs/Quintal)", "Reported Date"]
                
                driver.quit()
                return df # Return the data and exit the function
            else:
                print("Table found, but it is empty. Checking previous day.")

        except Exception:
            # This block will run if the wait times out (no table found)
            print("No data table found for this date. Checking previous day.")
            continue # Move to the next day in the loop

    print(f"\nNo data found for the last {search_days} days. The market is likely in the off-season.")
    driver.quit()
    return pd.DataFrame()

# --- Main execution block for EcoAdvisor to use ---
if __name__ == "__main__":
    latest_prices_df = get_latest_market_data()
    
    if not latest_prices_df.empty:
        print("\n--- Latest Available Cotton Prices in Gujarat (from agmarknet.gov.in) ---")
        print(latest_prices_df.head())
        
        # Save the retrieved data for further use
        output_file = "latest_agmarknet_cotton_prices.csv"
        latest_prices_df.to_csv(output_file, index=False)
        print(f"\nLatest available data successfully saved to '{output_file}'")
    else:
        print("\nCould not find any recent data on the portal.")
