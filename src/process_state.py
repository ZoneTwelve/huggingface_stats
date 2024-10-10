import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from tabulate import tabulate
from pathlib import Path

# Set locale for date formatting (optional depending on your system)
import locale
locale.setlocale(locale.LC_TIME, 'C')

def get_models_stats_df(
    files: str = "data/models",
    filetype: str = "csv",
):
    files = f"{files}/{'*.{}'.format(filetype)}"
    # Read all CSV files from the "data/models" directory
    data_files = glob.glob( files )

    # Load all CSVs into a single DataFrame
    df = pd.concat([pd.read_csv(file) for file in data_files])

    # Group by model name and date, and calculate total downloads
    df_model = (df.groupby(['model_name', 'date'])
                  .agg(downloads=('downloads', 'sum'))
                  .reset_index())

    # Extract year and month from the date
    df_model['date'] = pd.to_datetime(df_model['date'])
    df_model['year'] = df_model['date'].dt.year
    df_model['month'] = df_model['date'].dt.month.astype(str).str.zfill(2)

    # Arrange by model_name, year, and month
    df_model = df_model.sort_values(by=['model_name', 'year', 'month'])

    # Filter out unwanted model names
    df_model = df_model[~df_model['model_name'].str.contains("ModelCardReview|cp.|GGUF")]

    # Create a "year-month" column
    df_model['yearmonth'] = df_model['year'].astype(str) + "-" + df_model['month']
    return df_model, df

def main(files: str = "data/models", filetype: str = "csv"):
    df_model, _ = get_models_stats_df(files, filetype)
    
    # Get all-time maximum downloads for each model
    df_all = (_.groupby('model_name')
                 .agg(downloadsAllTime=('downloadsAllTime', 'max'))
                 .reset_index())

    df_all = df_all[~df_all['model_name'].str.contains("ModelCardReview|cp.|GGUF")]
    # Reshape the data for a table format, with downloads by month as columns
    df_table = df_model.pivot_table(index='model_name', 
                                    columns='yearmonth', 
                                    values='downloads', 
                                    fill_value=0)

    # Add the downloadsAllTime column
    df_table['downloadsAllTime'] = df_all.set_index('model_name')['downloadsAllTime']

    # Sort by downloadsAllTime
    df_table = df_table.sort_values(by='downloadsAllTime', ascending=False)

    # Reorder columns to move 'downloadsAllTime' to the front
    columns = ['downloadsAllTime'] + [col for col in df_table.columns if col != 'downloadsAllTime']
    df_table = df_table[columns]

    # Print the table in SQL-like format
    print(tabulate(df_table, headers='keys', tablefmt='psql'))
    
    # Print the table in markdown-like format
    print(tabulate(df_table.reset_index(), headers='keys', tablefmt='pipe'))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
