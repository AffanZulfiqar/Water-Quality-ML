import urllib.request
import urllib.parse
import pandas as pd
import io

base_url = 'https://www.waterqualitydata.us/data/Result/search'
params = {
    'statecode': 'US:53',
    'characteristicName': ['pH', 'Specific conductance', 'Turbidity', 'Sulfate', 'Total dissolved solids'],
    'sampleMedia': 'Water',
    'startDateLo': '01-01-2020',
    'startDateHi': '01-01-2024',
    'mimeType': 'csv',
    'zip': 'no'
}

query_string = urllib.parse.urlencode(params, doseq=True)
url = f"{base_url}?{query_string}"
print(f"Fetching from {url}")

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        content = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content), low_memory=False)
        print(f"Fetched {len(df)} rows.")
        print(df['CharacteristicName'].value_counts())
        df.to_csv('C:\\Users\\affan\\.gemini\\antigravity-ide\\scratch\\wqp_raw.csv', index=False)
except Exception as e:
    print(f"Error: {e}")
