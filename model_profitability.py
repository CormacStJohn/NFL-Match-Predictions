"""
Cormac St John
CSE 163 A26
Implements functions that create a
betting strategy using a model's
predictions.
"""


import data_cleaning
import schedule_formatting
import model
import polars as pl
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np


# A DataFrame of match schedules for the seasons of interest
SCHEDULES = data_cleaning.make_schedules(model.TRAINING_SEASONS)

# Columns of importance for calculating a model's profitability
TOTAL_COLUMNS = ["total_line", "under_odds", "over_odds", "total", "predicted_total"]
SPREAD_COLUMNS = ["spread_line", "away_spread_odds", "home_spread_odds",
                  "result", "predicted_diff"]

def create_predictions_df() -> pl.DataFrame:
    """
    Trains a Neural network on the first 1300 rows
    of schedule data that has been cleaned for training, then
    tests the model on 90 NFL matches that it was not trained
    on and records the model's predictions for the total score
    and the score difference in columns labeled "predicted_total"
    and "predicted_diff". Returns a DataFrame of the 90 matches
    with these two additional columns.
    """
    rosters = schedule_formatting.add_home_and_away(SCHEDULES)
    cleaned_rosters = data_cleaning.clean_for_training(rosters)
    train_data = cleaned_rosters.slice(0, 1300)
    test_data = cleaned_rosters.slice(1300, 90)

    trained_model, _ = model.train_model(train_data)

    test = torch.from_numpy(test_data.select(model.TRAINING_FEATURES)
                            .to_numpy().astype(np.float32, copy=False)).to("cuda")
    with torch.inference_mode():
        values = trained_model(test).cpu().numpy()

    return test_data.with_columns(
        pl.Series(name="predicted_diff", values=values[:, 2]),
        pl.Series(name="predicted_total", values=values[:, 3]),
    )

def get_total_profitability(data: pl.DataFrame) -> pl.DataFrame:
    """
    Takes a DataFrame data that was generated using
    create_predictions_df, and tests the profitability
    of the model's predictions of the total by implementing
    the following betting strategy:
    the model starts with $1000 and bets $50 on every game
    according to whether its predicted total is more or less
    than the total line. Its wealth is then modified according
    to the actual total value of the match. The function returns
    a copy of data with an additional column wealth storing the
    model's wealth after it has been updated according to the row's
    outcome.
    """
    money = 1000
    total_df = data[TOTAL_COLUMNS]
    wealth = []

    for total_line, under_odds, over_odds, total, predicted_total in total_df.iter_rows():
        if total == total_line:
            wealth.append(money)
            continue

        bet_over = predicted_total > total_line
        won = (bet_over and total > total_line) or (
            not bet_over and total < total_line
        )
        odds = over_odds if bet_over else under_odds

        if odds == 0:
            raise ValueError("Betting odds cannot be zero.")

        if won:
            if odds > 0:
                money += 50 * odds / 100
            elif odds < 0:
                money += 50 * 100 / abs(odds)
        else:
            money -= 50

        wealth.append(money)

    return data.with_columns(pl.Series(name="wealth", values=wealth))


def get_spread_profitability(data: pl.DataFrame) -> pl.DataFrame:
    """
    Takes a DataFrame data that was generated using
    create_predictions_df, and tests the profitability
    of the model's predictions of the spread
    by implementing the following betting strategy:
    the model starts with $1000 and bets $50 on every game
    according to whether its predicted spread is more or less
    than the spread line. Its wealth is then modified according
    to the actual spread of the match. The function returns
    a copy of data with an additional column wealth storing the
    model's wealth after it has been updated according to the row's
    outcome.
    """
    money = 1000
    spread_df = data[SPREAD_COLUMNS]
    wealth = []

    for spread_line, away_odds, home_odds, result, predicted_diff in spread_df.iter_rows():
        if result == spread_line:
            wealth.append(money)
            continue

        bet_home = predicted_diff > spread_line
        won = (bet_home and result > spread_line) or (
            not bet_home and result < spread_line
        )
        odds = home_odds if bet_home else away_odds

        if odds == 0:
            raise ValueError("Betting odds cannot be zero.")

        if won:
            if odds > 0:
                money += 50 * odds / 100
            elif odds < 0:
                money += 50 * 100 / abs(odds)
        else:
            money -= 50

        wealth.append(money)

    return data.with_columns(pl.Series(name="wealth", values=wealth, strict=False))


def plot_wealth(data: pl.DataFrame, ax: plt.axis):
    """
    Takes a DataFrame data with a column wealth as
    given in get_total_profitability or get_spread_profitability
    and an axis ax and creates a plot on ax of the model's wealth
    over time. The plot's x axis is labeled Match Number and its y
    axis is labeled Wealth.
    """
    matches = data["wealth"].len()
    sns.lineplot(x=[i for i in range(1, matches + 1)], y="wealth", data=data.to_pandas(), ax=ax)
    ax.set_xlabel("Match Number")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylabel("Wealth")


def main():
    predictions = create_predictions_df()

    spread_df = get_spread_profitability(predictions)
    total_df = get_total_profitability(predictions)

    _, ax = plt.subplots(1, 2)
    plot_wealth(spread_df, ax[0])
    plot_wealth(total_df, ax[1])
    ax[0].set_title("Spread Wealth Over Time")
    ax[1].set_title("Total Wealth Over Time")
    plt.savefig("wealth_plots.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
