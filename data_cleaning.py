"""
Cormac St John
CSE 163 A26
Implements functions which clean
nfl data and return dataframes
that are used as variables in
schedule_formatting.py and in
main.py
"""


import nflreadpy as nfl
import polars as pl


def make_schedules(seasons: int | list[int] | None) -> pl.DataFrame:
    """
    Takes an int, list of ints, or None seasons as an
    argument and returns a polars DataFrame of cleaned
    NFL schedule data from the specified seasons. If
    seasons is None, returns data from all seasons available.
    """
    cols_to_remove = ["away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
                      "weekday", "old_game_id", "nfl_detail_id", "pff", "gsis",
                      "pfr", "espn", "ftn", "div_game", "roof", "surface", "temp",
                      "wind", "referee", "stadium_id", "stadium", "gameday",
                      "away_coach", "home_coach"]
    df = nfl.load_schedules(seasons=seasons)

    return df.with_columns(df["gameday"].str.slice(0, 4).cast(int).alias("year"),
                           df["gameday"].str.slice(5, 2).cast(int).alias("month"),
                           df["gameday"].str.slice(8, 2).cast(int).alias("day")).drop(cols_to_remove)


def make_rankings() -> pl.DataFrame:
    """
    Returns a polars DataFrame of cleaned NFL fantasy
    ranking data.
    """
    cols_to_remove = ["scrape_date", "pos"]
    df = nfl.load_ff_rankings(type="all")

    return df.with_columns(df["scrape_date"].str.slice(0, 4).cast(int).alias("year"),
                           df["scrape_date"].str.slice(5, 2).cast(int).alias("month"),
                           df["scrape_date"].str.slice(8,2).cast(int).alias("day")).drop(cols_to_remove)


def clean_for_training(data: pl.DataFrame) -> pl.DataFrame:
    """
    Drops null values and modifies the columns of a
    DataFrame data containing match schedules to make the
    columns usable for training a machine learning model
    """
    df = data.drop_nulls().with_columns(pl.col("location").replace({"Home": 1,
                                                                    "Away": -1,
                                                                    "Neutral": 0}).cast(int))

    return df
