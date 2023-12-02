#!/usr/bin/env python3.11
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile


# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename: str) -> pd.DataFrame:
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27",
               "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t",
               "p5a"]

    # def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
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

    # Prázdný DataFrame pro načtení dat
    all_data = pd.DataFrame()

    # Načtení hlavního ZIP souboru
    with zipfile.ZipFile(filename, 'r') as z:
        print(f"Načítání hlavního ZIP souboru: {filename}")
        # Seznam všech ZIP souborů uvnitř hlavního ZIPu
        zip_files = z.namelist()

        for zip_file in zip_files:
            print(f"Zpracovávání ZIP souboru: {zip_file}")
            # Načtení každého ZIP souboru
            with z.open(zip_file) as inner_zip:
                with zipfile.ZipFile(inner_zip) as iz:
                    # Seznam všech CSV souborů uvnitř každého ZIPu
                    csv_files = [f for f in iz.namelist() if f.endswith('.csv') and not f.startswith('CHODCI')]

                    for csv_file in csv_files:
                        print(f"Zpracovávání souboru: {csv_file}")

                        # Ignorování prázdných souborů
                        if csv_file.startswith('08') or csv_file.startswith('09') or \
                                csv_file.startswith('10') or csv_file.startswith('11') or \
                                csv_file.startswith('12') or csv_file.startswith('13'):
                            print(f"Soubor {csv_file} je prázdný a bude vynechán.")
                            continue

                        # Získání kódu regionu z názvu souboru
                        region_code = csv_file.split('.')[0][-2:]
                        region = next((key for key, value in regions.items() if value == region_code), None)
                        if region is None:
                            print(f"Region code {region_code} not found in regions map.")
                            continue
                        print(f"Soubor {csv_file} patří do regionu: {region}")

                        # Načtení dat z CSV souboru
                        with iz.open(csv_file) as f:

                            data = pd.read_csv(f, encoding='cp1250', names=headers, sep=';', dtype=str)
                            data['region'] = region

                            # Sloučení s hlavním DataFrame
                            all_data = pd.concat([all_data, data], ignore_index=True)

        return all_data
    
# Ukol 2: zpracovani dat

def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        orig_size = sum(df.memory_usage(deep=True)) / 10 ** 6  # size in MB


    categorical_columns = [
        'p5a', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13a', 'p13b', 'p13c', 
        'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p27', 
        'p28', 'p29', 'p30', 'p31', 'p32', 'p33c', 'p33f', 'p33g', 'p35', 
        'p36', 'p37', 'p39', 'p44', 'p45a', 'p47', 'p48a', 'p49', 'p50a', 'p50b', 'p51', 'p52', 
        'p55a', 'p57', 'p58', 'weekday(p2a)', 'h', 'i', 'k', 'region'
    ]

    float_cols = ['a', 'b', 'd', 'e', 'f', 'g', 'o']

    # Convert 'p2a' to date format
    df['date'] = pd.to_datetime(df['p2a'])

    # Convert columns to categorical data type
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Convert specified columns to float
    for col in float_cols:
        # Replace commas with dots for float conversion
        df[col] = df[col].astype(str).str.replace(',', '.')
        # Convert to float, setting errors to 'coerce' to handle any non-convertible values
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicates based on 'p1'
    df = df.drop_duplicates(subset=['p1']) # Keep the first occurrence
    
    # Print the size information
    if verbose:
        print(f'orig_size={orig_size:.1f} MB')
        df = df.copy()
        new_size = sum(df.memory_usage(deep=True)) / 10 ** 6
        print(f'new_size={new_size:.1f} MB')

    return df

# Ukol 3: počty nehod oidke stavu řidiče
def plot_state(df, fig_location=None, show_figure=False):
    
    # Slovník pro mapování hodnot 'p57' na popisné texty
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

    # Aplikování mapování na sloupec 'p57'
    df['p57_text'] = df['p57'].map(driver_condition_mapping)

    # Agregace dat pro graf
    data_for_plotting = df.groupby(['region', 'p57_text'], observed=True).size().reset_index(name='pocet_nehod')

    # Determine the number of unique regions for the palette
    num_regions = df['region'].nunique()

    # Vytvoření figure-level grafu s seaborn
    g = sns.catplot(
        data=data_for_plotting, kind='bar',
        x='region', y='pocet_nehod', hue='region',
        col='p57_text', col_wrap=2, height=4, aspect=1.5,
        sharey=False, palette=sns.color_palette("hls", 14),
        
    )
    
    # Nastavení grafů a popisků, aby se nepřekrývaly
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Počet nehod")
    # Removing the rotation of x-tick labels
    g.set_xticklabels(rotation=0)

    # Nastavení pozadí pro každý podgraf
    for ax in g.axes.flatten():
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        # Upravení výšky y osy dle maximální hodnoty v každém podgrafu
        ax.set_ylim(0, data_for_plotting[data_for_plotting['p57_text'] == ax.get_title()]['pocet_nehod'].max())
        # Nastavení popisků pro x osu (regiony)
        ax.set_xticklabels(data_for_plotting['region'].unique(), rotation=0)

    # Uložení grafu do souboru, pokud je zadán fig_location
    if fig_location:
        g.fig.savefig(fig_location)
    
    # Zobrazení grafu, pokud je show_figure True
    if show_figure:
        plt.show()

# Ukol4: alkohol v jednotlivých hodinách


def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    pass


# Ukol 5: Zavinění nehody v čase


def plot_fault(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    pass

if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.
    df = load_data("data.zip")

    # Kontrola, zda sloupec 'o' existuje a zda obsahuje nějaké hodnoty
    #if 'o' in df.columns:
    #    # Filtrace hodnot ve sloupci 'o', které nejsou NaN
    #    non_null_o_values_1 = df['o'][df['o'].notna()]
    #
    #    # Výpis těchto hodnot
    #    print(non_null_o_values_1.head(10))
    #    
    #    # Nahrazení čárek tečkami pro konverzi na float
    #    df['o'] = df['o'].astype(str).str.replace(',', '.')
    #    # Převod na float, nastavení chyb na 'coerce' pro zpracování nekonvertovatelných hodnot
    #    df['o'] = pd.to_numeric(df['o'], errors='coerce')
    #
    #    non_null_o_values_2 = df['o'][df['o'].notna()]
    #    
    #    # Výpis těchto hodnot
    #    print(non_null_o_values_2.head(10))

    # Výběr první neprázdné hodnoty v každém sloupci
    #first_values = df.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else 'NaN')

    # Výpis hodnot tabulatorově
    #for column, value in first_values.items():
    #    print(f"{column}\t{value}")

    #df.to_pickle("dataframe.pkl")
    #df.to_csv('data.csv', index=False, sep=';', encoding='utf-8')
    #df = pd.read_pickle("dataframe.pkl")

    df2 = parse_data(df, True)

    #if 'o' in df2.columns:
    #    non_null_o_values_3 = df2['o'][df2['o'].notna()]
    #
    #    # Výpis těchto hodnot
    #    print(non_null_o_values_3.head(10))

    plot_state(df2, "01_state.png")
    #plot_alcohol(df2, "02_alcohol.png", True)
    #plot_fault(df2, "03_fault.png")

# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
