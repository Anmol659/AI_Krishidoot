import requests
import pandas as pd

def analyze_api_data():
    """
    Downloads the full data stream from the data.gov.in API, saves it,
    and analyzes it to find all unique commodity and state names.
    This is a diagnostic tool to find the correct names to use for filtering.
    """
    api_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b&format=json&offset=0&limit=10000"
    
    try:
        print("Requesting full historical dataset from data.gov.in...")
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        print("Data received. Processing...")
        
        if 'records' not in data or not data['records']:
            print("API response did not contain any data records.")
            return

        # Robustly parse all records
        cleaned_records = []
        for record in data['records']:
            cleaned_records.append({
                'state': record.get('state'),
                'commodity': record.get('commodity'),
                'market': record.get('market'),
                'arrival_date': record.get('arrival_date'),
                'modal_price': record.get('modal_price')
            })
        
        full_df = pd.DataFrame(cleaned_records)
        
        # Save the full, unfiltered data for inspection
        output_file = "unfiltered_agmarknet_data.csv"
        full_df.to_csv(output_file, index=False)
        print(f"\nFull unfiltered dataset saved to '{output_file}'")

        # --- DIAGNOSTIC ANALYSIS ---
        # Get unique commodity names and print them
        unique_commodities = full_df['commodity'].dropna().unique()
        print("\n--- Unique Commodity Names Found in the Database ---")
        for commodity in sorted(unique_commodities):
            print(commodity)
            
        # Get unique state names and print them
        unique_states = full_df['state'].dropna().unique()
        print("\n--- Unique State Names Found in the Database ---")
        for state in sorted(unique_states):
            print(state)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    analyze_api_data()