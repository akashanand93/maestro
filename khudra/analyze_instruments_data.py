import json
import pandas as pd

if __name__ == "main":
    instruments_file = "data/instruments.json"
    with open(instruments_file, 'r') as infile:
        instruments = json.load(infile)
    instruments_df = pd.DataFrame(instruments)
