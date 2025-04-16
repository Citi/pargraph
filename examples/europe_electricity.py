"""
Based on the monthly electricity production data from ENTSO-E, plots the percentage of renewable energy production for
the European electricity grid.

Usage:

    $ git clone https://github.com/Citi/parfun && cd parfun
    $ pip install -r examples/requirements
    $ python -m examples.europe_electricity [--plot]

"""

import sys
from typing import Iterable

import pandas as pd

from pargraph import delayed, graph


@delayed
def fetch_production_data(year: int) -> pd.DataFrame:
    """
    Downloads the monthly production data for the given year.

    Sourced from https://www.entsoe.eu/data/power-stats/.
    """

    url = f"https://www.entsoe.eu/publications/data/power-stats/{year}/monthly_domestic_values_{year}.csv"

    result = pd.read_csv(url, sep=r"\t|,|;", engine="python")

    # Some newer datasets use "Area" instead of "Country"
    if "Area" in result.columns:
        result["Country"] = result["Area"]

    return result[["Year", "Month", "Category", "Country", "ProvidedValue"]]


@delayed
def make_consumption_negative(production_data: pd.DataFrame) -> pd.DataFrame:
    """
    Make consumption values negative production values.

    Some production categories have positive consumption values (e.g. "Consumption of Hydro Water Reservoir"). This
    function transforms these values in their production counter parts, but with a negative value. This simplifies
    subsequent processing.
    """

    PREFIX = "Consumption of "

    result = production_data.copy()

    is_consumption = result["Category"].str.startswith(PREFIX)

    result.loc[is_consumption, "Category"] = result.loc[is_consumption, "Category"].str.replace(PREFIX, "", regex=False)
    result.loc[is_consumption, "ProvidedValue"] *= -1

    return result


@delayed
def group_production_by_type(production_data: pd.DataFrame) -> pd.DataFrame:
    """Groups and sums all production data by type ("Fossil", "Nuclear", "Renewable" and "Other")."""

    fossil_sources = {
        "Fossil Gas", "Fossil Hard coal", "Fossil Oil",
        "Fossil Brown coal/Lignite", "Fossil Coal-derived gas",
        "Fossil Oil shale", "Fossil Peat",
    }
    nuclear_sources = {"Nuclear"}
    renewable_sources = {
        "Biomass", "Solar", "Wind Onshore", "Wind Offshore", "Geothermal",
        "Hydro Pumped Storage", "Hydro Run-of-river and poundage",
        "Hydro Water Reservoir", "Marine", "Other renewable",
    }

    def map_category(category: str) -> str:
        if category in fossil_sources:
            return "Fossil"
        elif category in nuclear_sources:
            return "Nuclear"
        elif category in renewable_sources:
            return "Renewable"
        else:
            return "Other"

    result = production_data.copy()

    result["EnergyType"] = result["Category"].map(map_category)
    del result["Category"]

    return result.groupby(["Year", "Month", "EnergyType"])["ProvidedValue"].sum().reset_index()


@delayed
def monthly_percentage_production(production_data: pd.DataFrame) -> pd.DataFrame:
    """Returns the monthly production percentage for every month and every energy source type."""

    result = production_data.pivot_table(index=["Year", "Month"], columns="EnergyType", values="ProvidedValue")

    result = result.div(result.sum(axis=1), axis=0) * 100  # make it percentages

    # Uses datetime for year-month
    result.index = pd.to_datetime({
        "year": result.index.get_level_values(0),
        "month": result.index.get_level_values(1),
        "day": 1,
    })

    result.sort_index(ascending=True)  # sort by date

    return result


@delayed
def concat(*dataframes: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dataframes)


@graph
def get_monthly_percentage_production(years: Iterable[int]) -> pd.DataFrame:
    processed_yearly_data = []
    for year in years:
        yearly_production_data = fetch_production_data(year)

        yearly_production_data = make_consumption_negative(yearly_production_data)
        yearly_production_data = group_production_by_type(yearly_production_data)
        yearly_production_data = monthly_percentage_production(yearly_production_data)

        processed_yearly_data.append(yearly_production_data)

    return concat(*processed_yearly_data)


def plot_electricity_production(production_percentages: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "Fossil": "lightcoral",
        "Nuclear": "violet",
        "Renewable": "lightgreen",
        "Other": "lightsteelblue",
    }

    production_percentages.index = production_percentages.index.strftime("%b %Y")
    production_percentages.plot(kind="bar", stacked=True, figsize=(10, 6), width=1, color=colors)

    plt.title("Europe's monthly electricity production by source")
    plt.ylabel("Percentage (%)")
    plt.xlabel('Month')
    plt.legend(title="Energy source", loc='upper left')
    plt.grid(axis="y", linestyle="--")
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()


def main():
    YEARS = range(2019, 2025)

    processed_data = get_monthly_percentage_production(YEARS)  # this works

    # from pargraph import GraphEngine
    # graph_engine = GraphEngine()
    # task_graph, keys = get_monthly_percentage_production.to_graph().to_dict(years=YEARS)  # this blocks forever
    # processed_data = graph_engine.get(task_graph, keys)[0]

    if "--plot" in sys.argv[1:]:
        plot_electricity_production(processed_data)
    else:
        print(processed_data)


if __name__ == "__main__":
    main()
