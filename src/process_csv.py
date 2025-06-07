import pandas as pd
import os
from datetime import datetime

def process_csv(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df = pd.read_csv(input_file)
    
    # Filter for rated games only
    df = df[df['rated'] == True]
    
    # Only keep games with a valid winner
    df = df[df['winner'].isin(['white', 'black', 'draw'])]
    
    # Map winner to result
    result_map = {'white': 1.0, 'black': 0.0, 'draw': 0.5}
    df['result'] = df['winner'].map(result_map)
    
    # Convert created_at (ms since epoch) to UTC datetime string
    df['timestamp'] = pd.to_datetime(df['created_at'], unit='ms', utc=True)
    
    # Select and reorder columns
    out_df = df[['white_id', 'black_id', 'result', 'timestamp']]
    
    out_df.to_csv(output_file, index=False)
    print(f"Processed {len(out_df)} games. Data saved to {output_file}")

if __name__ == "__main__":
    input_file = "./data/raw/games.csv"
    output_file = "./data/processed/games_clean.csv"
    process_csv(input_file, output_file) 