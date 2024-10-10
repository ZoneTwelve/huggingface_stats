import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import glob
import os
from process_state import get_models_stats_df

def render_plot(
    df: DataFrame,
    return_plot: bool = True,
):
    
    # Plot the download trends for each model
    plt.figure(figsize=(12, 8))

    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        plt.plot(model_data['yearmonth'], model_data['downloads'], label=model)

    # Customize the plot
    plt.xlabel('Year-Month')
    plt.ylabel('Downloads')
    plt.title('Download Trends by Model Over Time')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if return_plot:
        return plt
    # Display the plot
    plt.show()

def main(
    files: str = "data/models",
    filetype: str = "csv",
):
    df, _ = get_models_stats_df(files, filetype)
    print(df)
    render_plot(df, return_plot=False)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
