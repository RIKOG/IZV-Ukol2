#!/usr/bin/env python3.11
# coding=utf-8
# Author: Richard Gajdosik, xgajdo33

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from a ZIP file located at the specified filename.

    This function reads ZIP files for each year, where data are divided by regions.
    Data in each CSV file are then combined into a single DataFrame. A new column 'region'
    containing three-letter abbreviations of each region is added to the DataFrame.

    Parameters:
    - filename: Path to the ZIP file containing the data.

    Returns:
    - A pandas DataFrame containing the loaded data with an additional 'region' column.

    Note: The function assumes data in cp1250 encoding and specific column names as defined
    in the project description.
    """

    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27",
               "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t",
               "p5a"]

    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    # Empty DataFrame to load data
    all_data = pd.DataFrame()

    # Loading the main ZIP file
    with zipfile.ZipFile(filename, 'r') as z:

        # Save all ZIP files inside the main ZIP
        zip_files = z.namelist()

        for zip_file in zip_files:

            # Processing each ZIP file
            with z.open(zip_file) as inner_zip:
                with zipfile.ZipFile(inner_zip) as iz:

                    # List of all CSV files inside each ZIP not counting walkers
                    csv_files = [f for f in iz.namelist() if f.endswith('.csv') and not f.startswith('CHODCI')]

                    for csv_file in csv_files:

                        # Ignoring empty files
                        if any(csv_file.startswith(prefix) for prefix in ['08', '09', '10', '11', '12', '13']):
                            continue

                        # Extracting region code from the file name
                        region_code = csv_file.split('.')[0][-2:]
                        region = next((key for key, value in regions.items() if value == region_code), None)
                        if region is None:
                            continue

                        # Loading data from CSV file
                        with iz.open(csv_file) as f:

                            data = pd.read_csv(f, encoding='cp1250', names=headers, sep=';', dtype=str)
                            data['region'] = region

                            # Merging with the main DataFrame
                            all_data = pd.concat([all_data, data], ignore_index=True)

        return all_data

# Ukol 2: zpracovani dat


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Parse and clean the data from the DataFrame obtained by calling load_data().

    This function converts the 'p2a' column to a datetime format, categorizes suitable columns,
    converts specified columns to float or integer types,
    and duplicate records based on the 'p1' identifier are removed.

    Parameters:
    - df: DataFrame obtained from load_data() function.
    - verbose: If True, prints the size of the DataFrame before and after parsing.

    Returns:
    - A cleaned and parsed DataFrame.

    The function targets to reduce the memory usage below 0.7 GB.

    """
    if verbose:
        orig_size = sum(df.memory_usage(deep=True)) / 10 ** 6  # size in MB

    # Keeping everything categorical because of it not being needed in later tasks
    categorical_columns = [
        'p5a', 'p6', 'p7', 'p8', 'p9', 'p12', 'p13a', 'p13b', 'p13c', 'p14',
        'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p27',
        'p28', 'p33c', 'p33f', 'p33g', 'p34', 'p35',
        'p36', 'p37', 'p39', 'p44', 'p45a', 'p47', 'p48a', 'p49', 'p50a', 'p50b', 'p51', 'p52',
        'p53', 'p55a', 'p57', 'p58', 'weekday(p2a)', 'h', 'i', 'k', 'region'
    ]

    float_cols = ['a', 'b', 'd', 'e', 'f', 'g', 'o']
    int_cols = ['p2b', 'p11', 'p10']

    # Convert 'p2a' to date format
    df['p2a'] = pd.to_datetime(df['p2a'])

    # Convert columns to categorical data type
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Convert specified columns to float
    for col in float_cols:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert specified columns to integer
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(pd.Int32Dtype())

    # Remove duplicates based on 'p1'
    df = df.drop_duplicates(subset=['p1'])

    # Print the size information
    if verbose:
        df = df.copy()
        new_size = sum(df.memory_usage(deep=True)) / 10 ** 6
        print(f'orig_size={orig_size:.1f} MB')
        print(f'new_size={new_size:.1f} MB')

    return df

# Ukol 3: počty nehod podle stavu řidiče


def plot_state(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Create a bar graph of the number of accidents in different regions based on the state of drivers.

    The function processes the provided DataFrame to plot the number of accidents for each
    driver's state across different regions. It consists of 6 subplots arranged in a grid of 3 rows and 2 columns.

    Parameters:
    - df: DataFrame outputted from the parse_data function.
    - fig_location: Path to save the figure.
    - show_figure: Whether to display the figure or not.

    The function uses column 'p57' to differentiate states of drivers, identified by numbers 1 to 9.
    It also sets proper titles, axis labels, and custom background for each subplot.
    """
    # Dictionary to map 'p57' values to descriptive texts
    driver_condition_mapping = {
        '1': 'Stav řidiče: dobrý',
        '2': 'Stav řidiče: unaven, usnul, náhlá fyzická indispozice',
        '3': 'Stav řidiče: pod vlivem léků, narkotik',
        '4': 'Stav řidiče: pod vlivem alkoholu, do 0,99 ‰',
        '5': 'Stav řidiče: pod vlivem alkoholu, 1‰ a více',
        '6': 'Stav řidiče: nemoc, úraz apod.',
        '7': 'Stav řidiče: invalida',
        '8': 'Stav řidiče: řidič při jízdě zemřel',
        '9': 'Stav řidiče: pokus o sebevraždu, sebevražda'
    }

    df = df.copy()

    # Exclude rows where 'p57' is NaN or '0' before mapping
    df = df[df['p57'].notna() & (df['p57'] != '0') & (df['p57'] != '1') & (df['p57'] != '2') & (df['p57'] != '3')]

    # Apply mapping to 'p57' column
    df['p57_text'] = df['p57'].map(driver_condition_mapping)

    # Aggregate data for plotting
    data_for_plotting = df.groupby(['region', 'p57_text'], observed=True).size().reset_index(name='pocet_nehod')

    # Create a figure-level plot with seaborn
    g = sns.catplot(
        data=data_for_plotting, kind='bar',
        x='region', y='pocet_nehod', hue='region',
        col='p57_text', col_wrap=2, height=4, aspect=1.5,
        sharey=False, palette=sns.color_palette("hls", 14),
        legend=False,
    )

    # Adjust plots and labels to avoid overlapping
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Počet nehod")
    g.set_xticklabels(rotation=0)
    g.fig.suptitle("Počet nehod dle stavu řidiče při nedobrém stavu", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Set the background color for each subplot
    for ax in g.axes.flatten():
        ax.set_facecolor('#DDEEFF')
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        ax.set_ylim(0, data_for_plotting[data_for_plotting['p57_text'] == ax.get_title()]['pocet_nehod'].max())
        ax.set_xticklabels(data_for_plotting['region'].unique(), rotation=0)

    # Iterate over the axes to set the x-axis label for the bottom-most plots
    for ax in g.axes[-2:]:
        ax.set_xlabel('Kraj')

    # Hide x-axis labels for all other plots
    for ax in g.axes[:-2]:
        ax.set_xlabel('')

    # Save the plot if a file location is provided
    if fig_location:
        plt.savefig(fig_location)

    # Show the plot if requested
    if show_figure:
        plt.show()

# Ukol4: alkohol v jednotlivých hodinách


def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Create a bar graph showing the number of accidents involving alcohol in different hours for selected regions.

    The function processes the provided DataFrame to plot the number of accidents involving alcohol
    or not, across different hours of the day for four selected regions. The accidents without a known time are excluded.

    Parameters:
    - df: DataFrame outputted from the parse_data function.
    - fig_location: Path to save the figure.
    - show_figure: Whether to display the figure or not.

    The function uses column 'p2b' for the hour and 'p11' for alcohol involvement. 
    It aggregates data using groupby and plots using seaborn's figure-level plotting.
    """
    df = df.copy()

    # Assuming 'p2b' is in HHMM integer format
    df['Hour'] = (df['p2b'] // 100).astype(pd.Int32Dtype())
    # Only keep valid hours
    df = df[df['Hour'].between(0, 23)]

    # Determine if alcohol was involved
    df['Alkohol'] = df['p11'].apply(lambda x: 'Ano' if x in [1, 3, 5, 6, 7, 8, 9] else 'Ne')

    # Aggregate data
    grouped = df.groupby(['region', 'Hour', 'Alkohol'], observed=True).size().reset_index(name='Počet nehod')

    # Create the plot
    g = sns.catplot(
        data=grouped,
        x='Hour',
        y='Počet nehod',
        hue='Alkohol',
        col='region',
        kind='bar',
        height=5,
        aspect=1.5,
        sharex=False,
        sharey=False,
        col_wrap=2,
        legend=True
    )

    # Update the titles and labels
    g.set_titles("Kraj: {col_name}")
    g.set_axis_labels("Hodina", "Počet nehod")

    # Adjust the layout
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle('Alkohol v jednotlivých hodinách', fontsize=16)

    # Save the plot if a file location is provided
    if fig_location:
        plt.savefig(fig_location)

    # Show the plot if requested
    if show_figure:
        plt.show()

# Ukol 5: Zavinění nehody v čase


def plot_fault(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Create a line graph showing the number of accidents caused by different entities over time for selected regions.

    The function processes the provided DataFrame to plot the number of accidents caused by drivers of motor
    or non-motor vehicles, pedestrians, and animals for four selected regions over different months.
    The data is aggregated monthly and displayed in a line chart.

    Parameters:
    - df: DataFrame outputted from the parse_data function.
    - fig_location: Path to save the figure.
    - show_figure: Whether to display the figure or not.

    The function uses columns 'p2a' for the date and 'p10' for the cause of the accident.
    The x-axis is limited from January 1, 2016, to January 1, 2023. The resulting plots are adjusted for clarity.
    """
    df = df.copy()

    color_palette = {
        'řidičem motorového vozidla': 'red',
        'řidičem nemotorového vozidla': 'blue',
        'chodcem': 'green',
        'lesní zvěří, domácím zvířectvem': 'orange'
    }

    # Select four regions and causes
    selected_regions = ['PHA', 'STC', 'JHC', 'PLK']
    causes = {
        1: 'řidičem motorového vozidla',
        2: 'řidičem nemotorového vozidla',
        3: 'chodcem',
        4: 'lesní zvěří, domácím zvířectvem'
    }

    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, region in enumerate(selected_regions):

        mask = (df['region'] == region) & (df['p10'].isin(causes.keys()))

        df.loc[mask, 'Cause'] = df.loc[mask, 'p10'].map(causes)
        df_region = df[mask].copy()

        df_region['Cause'] = df_region['p10'].map(causes)

        pivot = df_region.pivot_table(index='p2a', columns='Cause', values='p1', aggfunc='count', fill_value=0)
        monthly_data = pivot.resample('M').sum().stack().reset_index(name='Počet nehod')
        monthly_data['date'] = monthly_data['p2a'].dt.strftime('%Y-%m')

        # Plotting the data with the new color palette
        sns.lineplot(data=monthly_data, x='date', y='Počet nehod',
                     hue='Cause', palette=color_palette, ax=axs[i])

        # Customizing the ticks and labels
        axs[i].set_xticks([f'{year}-01' for year in range(2016, 2024)])
        axs[i].set_xticklabels([f'{year}' for year in range(2016, 2024)])
        axs[i].set_title(f'Kraj: {region}')
        axs[i].grid(True)
        axs[i].set_xlabel('Datum')
        # Remove the legend from each subplot
        axs[i].get_legend().remove()

    # Set the common labels and title
    fig.suptitle('Zavinění nehody v čase podle regionu', fontsize=16)
    plt.xlabel('Datum')
    plt.ylabel('Počet nehod')

    # Adjust the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create a single legend for the whole figure
    handles, labels = axs[-1].get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center',
               ncol=len(causes), bbox_to_anchor=(0.5, 0.01), frameon=False)

    # Save the plot if a file location is provided
    if fig_location:
        plt.savefig(fig_location)

    # Show the plot if requested
    if show_figure:
        plt.show()


if __name__ == "__main__":

    df = load_data("data.zip")
    # df.to_pickle("dataframe.pkl")
    # df = pd.read_pickle("dataframe.pkl")
    df2 = parse_data(df, True)
    # df.to_csv('data.csv', index=False, sep=';', encoding='utf-8')
    plot_state(df2, "01_state.png")
    plot_alcohol(df2, "02_alcohol.png")
    plot_fault(df2, "03_fault.png")
