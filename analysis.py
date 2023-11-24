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
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

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
    
    float_cols = ["p37", "a", "b", "d", "e", "f", "g", "o", "n", "r", "s"]
    
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
                            data = pd.read_csv(f, encoding='cp1250', names=headers, sep=';', low_memory=False)
                            data['region'] = region
                            
                            # Sloučení s hlavním DataFrame
                            all_data = pd.concat([all_data, data], ignore_index=True)
                            
        # Resetování indexu a odstranění výchozího indexu
        all_data.reset_index(drop=True, inplace=True)
        return all_data

# Ukol 2: zpracovani dat


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    pass

# Ukol 3: počty nehod oidke stavu řidiče


def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    pass

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
    # Uložení dat do souboru Excel nebo CSV
    df.to_csv('cesta_k_souboru.csv', index=False, sep=';', encoding='utf-8')
    df2 = parse_data(df, True)

    plot_state(df2, "01_state.png")
    plot_alcohol(df2, "02_alcohol.png", True)
    plot_fault(df2, "03_fault.png")


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
