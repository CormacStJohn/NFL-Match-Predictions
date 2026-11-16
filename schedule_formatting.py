"""
Cormac St John
CSE 163 A26
Implements a function that
adds ranking data for the home
or away team of a given NFL match.
This function is used in the creation
of the training and testing data.
"""


import nflreadpy as nfl
import polars as pl
import data_cleaning


# A list of abbreviations for positions on a football team.
POSITIONS: list[str] = ["WR", "DB", "DL", "TE", "K", "LB", "RB", "QB"]


# A dictionary consisting of abbreviations for positions, as 
# well as the number of players on the field that have that
# position at a given time.
POSITION_AMOUNTS: dict[str, int] = {"WR": 2, "OL": 5, "DB": 4, "DL": 4,
                                    "TE": 1, "K": 1, "LB": 3, "LS": 1,
                                    "RB": 2, "P": 1, "QB": 1}


# A list of the columns that are relevant for the add_roster
# function
R_COLS = ["home_team", "away_team",
                   "year", "month", "day", "week", "season"]


# Schemas for the columns that will be added to the DataFrame of
# a given NFL match.
HOME_R_SCHEMA = pl.Struct({
    f"home_{position}_{i}": pl.Float64
    for position in POSITIONS
    for i in range(POSITION_AMOUNTS[position])
})
AWAY_R_SCHEMA = pl.Struct({
    f"away_{position}_{i}": pl.Float64
    for position in POSITIONS
    for i in range(POSITION_AMOUNTS[position])
})


def add_roster(game: pl.dataframe, home: bool) -> pl.dataframe:
    """
    Takes a game given by a row in the schedules dataframe
    and a boolean home (true for home team, false for away
    team), and adds columns for each position (accounting
    for the number of each position on a team, so there will
    be 2 wide receiver columns, 5 offensive line columns, etc),
    with the value in that column corresponding to the fantasy
    ranking of the highest ranked player on the team with that
    position (for positions with multiple players, the highest
    ranked player goes in the leftmost column of that position,
    the second highest ranked in the second leftmost column, etc).
    """

    # Getting relevant NFL data
    weekly_rosters = nfl.load_rosters_weekly(seasons=True)
    rankings = data_cleaning.make_rankings()
    
    output = {}
    is_home = "home" if home else "away"
    if home:
        team = game["home_team"]
    else:
        team = game["away_team"]

    # Storing game date info
    year = game["year"]
    month = game["month"]
    day = game["day"]

    # Getting the active roster of the chosen team for the
    # given game
    act_roster = weekly_rosters.filter((weekly_rosters["team"] == team) &
                                       (weekly_rosters["status"] == "ACT") &
                                       (weekly_rosters["week"] == game["week"]) &
                                       (weekly_rosters["season"] == game["season"]))

    # Getting the rankings of the active roster near the
    # date the game took place
    rankings = rankings
    act_names = act_roster["full_name"].to_list()
    year_mask = (rankings["year"] == year)
    month_mask = (rankings["month"] == month)
    date_mask = rankings["day"] <= day
    name_mask = rankings["player"].is_in(act_names)
    act_rankings = rankings.filter(year_mask & month_mask & date_mask & name_mask)

    # filtering the scrape dates to only data
    # from the scrape that took place closest
    # to the time of the game. This only works
    # if there is data in the same month as the
    # game from a date less than or equal to 
    # the game's date, so we check for that
    # first
    if len(act_rankings.rows()) != 0:
        date_mask2 = act_rankings["day"] == act_rankings["day"].unique().max()
        act_rankings = act_rankings.filter(date_mask2)

    # If there is no scraped data from a date 
    # in the same month of the game before 
    # the game took place, we grab the most
    # recent previous scrape
    else:
        alternate_month_mask = rankings["month"] <= month
        alternate_year_mask = rankings["year"] <= year
        month_year_filtered = rankings.filter(alternate_year_mask & alternate_month_mask)

        max_year_mask = month_year_filtered["year"] == month_year_filtered["year"].max()
        month_year_filtered = month_year_filtered.filter(max_year_mask)
        max_month_mask = month_year_filtered["month"] == month_year_filtered["month"].max()
        month_year_filtered = month_year_filtered.filter(max_month_mask)
        max_date_mask = month_year_filtered["day"] == month_year_filtered["day"].max()
        act_rankings = month_year_filtered.filter(max_date_mask)

    # adding the positions of the active roster as a
    # column of the rankings dataframe
    act_rankings = act_rankings.join(act_roster["full_name", "position"],
                                     how="left",
                                     left_on="player",
                                     right_on="full_name")

    # Sorting the active players by ecr ranking
    act_rankings = act_rankings.sort("ecr")

    for position in POSITIONS:
        for i in range(POSITION_AMOUNTS[position]):
            position_mask = act_rankings["position"] == position
            position_value = act_rankings.filter(position_mask)["ecr"].slice(i, 1)
            # If there does not exist a player with the specified position
            # we set their ranking to null
            if len(position_value) == 0:
                position_value = pl.Series(values=[None])
            output[f"{is_home}_{position}_{i}"] = position_value.item()

    return output


def add_home_and_away(data: pl.DataFrame) -> pl.DataFrame:
    """
    Takes a DataFrame data of NFL match schedules
    and returns a copy of it with columns of the
    home and away team's fantasy ranking data as
    provided by add_roster. To save time, the function
    saves the output on its first call, and reuses
    that output on future calls
    """
    try:
        return pl.read_csv("schedules.csv")
    except:
        h_cols = data[R_COLS].to_struct().map_elements(lambda x: add_roster(x, True)).alias("h")
        a_cols = data[R_COLS].to_struct().map_elements(lambda x: add_roster(x, False)).alias("a")

        df = data.with_columns(h_cols, a_cols).unnest(["h", "a"])
        df.write_csv("schedules.csv")
        return df
