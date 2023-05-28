# Datagen Pipeline

## 1. data_collector.py

- Dest folder: `raw_data`.
- This downloads data from Kite.
- Data is in `json` format.
- For each instrument, there is a separate file.

## 2. json_to_csv.py

- Src folder: `raw_data`.
- Dest folder: `csv_data`.
- This converts raw `json` files to `csv` files.
- Stocks with more missing values are removed.
- All stocks are clubbed into a single `csv` file.

## 3. transform.py : LTSF

- Src folder: `csv_data`.
- Dest folder: `ltsf`.
- This preprocesses the `csv` file for `LTSF` model.
- It drops stocks having certain missing values.
- It interpolates missing values.