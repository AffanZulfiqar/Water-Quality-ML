import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
log = logging.getLogger(__name__)

def main():
    log.info("Loading raw WQP data...")
    df_raw = pd.read_csv('data/wqp_raw.csv', low_memory=False)
    log.info(f"Loaded {len(df_raw)} raw measurements.")

    # Select relevant columns
    cols = ['ActivityIdentifier', 'CharacteristicName', 'ResultMeasureValue', 'ResultMeasure/MeasureUnitCode']
    df = df_raw[cols].copy()

    # Clean numeric values (some values might have < or ND)
    df['ResultMeasureValue'] = pd.to_numeric(df['ResultMeasureValue'], errors='coerce')
    df = df.dropna(subset=['ResultMeasureValue'])
    log.info(f"Remaining after numeric conversion: {len(df)} measurements.")

    # Map parameters
    param_map = {
        'pH': 'ph',
        'Specific conductance': 'Conductivity',
        'Turbidity': 'Turbidity',
        'Sulfate': 'Sulfate',
        'Total dissolved solids': 'Solids'
    }
    df['Feature'] = df['CharacteristicName'].map(param_map)

    # Pivot
    log.info("Pivoting data to wide format...")
    df_pivot = df.pivot_table(index='ActivityIdentifier', columns='Feature', values='ResultMeasureValue', aggfunc='mean').reset_index()

    req_features = ['ph', 'Conductivity', 'Turbidity', 'Sulfate', 'Solids']
    
    # Filter out missing
    df_clean = df_pivot.dropna(subset=req_features)
    log.info(f"Final overlapping samples with all 5 features: {len(df_clean)}")

    df_clean.to_csv('data/wqp_clean.csv', index=False)
    log.info("Saved to data/wqp_clean.csv")

if __name__ == '__main__':
    main()
