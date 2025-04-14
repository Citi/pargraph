import pandas as pd

from pargraph import GraphEngine, delayed, graph


@delayed
def get_total_amount_sum(file_url, unit):
    """
    Calculate the sum of the 'total_amount' column from a Parquet file and divide it by a given unit.

    :param file_url: The URL of the Parquet file
    :param unit: The unit to divide the total amount sum by
    :return: The sum of the 'total_amount' column divided by the given unit
    """
    return pd.read_parquet(file_url, columns=["total_amount"])["total_amount"].sum() / unit


@delayed
def add(*args):
    """
    Calculate the sum of all the arguments provided.

    :param args: A variable number of arguments to be summed
    :return: The sum of all arguments
    """
    return sum(args)


@delayed
def collect_result(*results):
    """
    Collect results, using every other element as the index and the remaining elements as the values.

    :param results: Arguments where even-indexed elements are used as the index and odd-indexed elements as the values
    :return: A pandas Series with the specified index and values
    """
    return pd.Series(results[1::2], index=results[0::2])


URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet"


@graph
def nyc_yellow_taxi_fare_total_for_year(year, unit):
    """
    Calculate the total fare amount for a given year by summing the total amounts from each month.

    :param year: The year for which to calculate the total fare amount
    :param unit: The unit to divide the total amount sum by
    :return: Total fare amount for the given year
    """
    return add(*(get_total_amount_sum(URL.format(year=year, month=month), unit) for month in range(1, 13)))


@graph
def nyc_yellow_taxi_fare_total_by_year(years, unit):
    """
    Calculate the total fare amount for multiple years.

    :param years: A list of years for which to calculate the total fare amount
    :param unit: The unit to divide the total amount sum by
    :return: Total fare amounts for the given years
    """
    results = []

    for year in years:
        results.append(year)
        results.append(nyc_yellow_taxi_fare_total_for_year(year, unit))

    return collect_result(*results)


if __name__ == "__main__":
    task_graph, keys = nyc_yellow_taxi_fare_total_by_year.to_graph(years=range(2024, 2025)).to_dict(unit=1_000_000)

    graph_engine = GraphEngine()
    print(graph_engine.get(task_graph, keys)[0])
