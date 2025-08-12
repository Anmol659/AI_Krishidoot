import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import os

MAIN_URL = "https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx"
PART2_URL = "https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReportpart2.aspx"
OUTPUT_FILE = "gujarat_cotton_prices.csv"
HTML_SAVE_PATH = "market_data.html"

def get_form_fields(soup):
    return {inp.get('name'): inp.get('value', '') for inp in soup.find_all('input') if inp.get('name')}

def get_dropdown_value(soup, select_id, visible_text):
    sel = soup.find('select', {'id': select_id})
    for opt in sel.find_all('option'):
        if opt.text.strip().lower() == visible_text.lower():
            return opt.get('value')
    raise ValueError(f"Option {visible_text} not found")

def fetch_html():
    session = requests.Session()

    # Step 1: Load main page to grab hidden form fields
    r1 = session.get(MAIN_URL, timeout=30)
    soup1 = BeautifulSoup(r1.text, 'html.parser')
    form = get_form_fields(soup1)

    form['ddlCommodity'] = get_dropdown_value(soup1, 'ddlCommodity', 'Cotton')
    form['ddlState'] = get_dropdown_value(soup1, 'ddlState', 'Gujarat')
    
    # For safety, you can choose a small recent range like last 7 days
    today = datetime.today()
    form['txtFromDate'] = (today - timedelta(days=7)).strftime("%d-%b-%Y")
    form['txtToDate'] = today.strftime("%d-%b-%Y")
    form['btnExportToExcel'] = 'Export To Excel'

    # Step 2: Submit to part2.aspx to get the actual table embedded in HTML
    r2 = session.post(PART2_URL, data=form, timeout=30)
    r2.raise_for_status()

    # Save to disk as .html if you like
    with open(HTML_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write(r2.text)

    return r2.text

def parse_html_for_data(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    table = soup.find('table', {'class': 'tableagmark_new'})
    if not table:
        print("⚠ No data table found.")
        return pd.DataFrame()

    return pd.read_html(str(table))[0]

def update_dataset():
    print("Fetching HTML from part2.aspx...")
    html = fetch_html()

    df_new = parse_html_for_data(html)
    if df_new.empty:
        print("No data extracted.")
        return

    df_new.columns = [col.strip() for col in df_new.columns]
    cols_to_keep = ['Market', 'Minimum Price(Rs./Quintal)']
    df_filtered = df_new[cols_to_keep] if all(c in df_new.columns for c in cols_to_keep) else df_new

    if os.path.exists(OUTPUT_FILE):
        df_old = pd.read_csv(OUTPUT_FILE)
    else:
        df_old = pd.DataFrame(columns=df_filtered.columns)

    df_combined = pd.concat([df_old, df_filtered], ignore_index=True).drop_duplicates()
    df_combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Updated {OUTPUT_FILE} — total rows now: {len(df_combined)}")

if __name__ == '__main__':
    update_dataset()
