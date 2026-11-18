"""
Cormac St John
CSE 163 A26
Tests the functions in schedule_formatting
and data_cleaning, and creates visualizations
that illustrate potential model overfitting at
different epoch numbers.
"""


import data_cleaning
import schedule_formatting as sf
import model
import model_profitability
import polars as pl
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def test_schedule_formatting() -> None:
    """
    Tests the add_roster, add_home_and_away,
    and get_all_schedules functions within the
    schedule_formatting file.
    """

    schedule = data_cleaning.make_schedules(2020).slice(0, 1)
    h_cols = schedule[sf.R_COLS].to_struct().map_elements(
            lambda x: sf.add_roster(x, True)).alias("h")
    schedule_with_home = schedule.with_columns(h_cols).unnest("h")

    # Stores the training features of the home team
    training_features1 = set(model.TRAINING_FEATURES[5:23])

    assert (set(schedule_with_home.columns) & training_features1)\
        == training_features1
    print("add_roster test passed!")

    schedule_h_and_a = sf.add_home_and_away(schedule)

    # Stores all training features
    training_features2 = set(model.TRAINING_FEATURES[5:])

    assert (set(schedule_h_and_a.columns) & training_features2)\
        == training_features2
    print("add_home_and_away tests passed!")

    seasons = set(model.TRAINING_SEASONS)
    all_schedules = sf.get_all_schedules()
    assert set(all_schedules["season"].unique().to_list()) & seasons == seasons
    print("get_all_schedules tests passed!")

    print("schedule_formatting tests passed!")


def test_data_cleaning() -> None:
    """
    Tests the make_schedules, make_rankings,
    and clean_for_training methods within the
    data_cleaning file
    """

    schedules = data_cleaning.make_schedules(None)
    schedules_popped_columns = \
        set(["away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
             "weekday", "old_game_id", "nfl_detail_id", "pff", "gsis",
             "pfr", "espn", "ftn", "div_game", "roof", "surface", "temp",
             "wind", "referee", "stadium_id", "stadium", "gameday",
             "away_coach", "home_coach"])
    schedules_added_columns = set(["year", "month", "day"])
    assert schedules_popped_columns & set(schedules.columns) == set()
    assert len(schedules_added_columns & set(schedules.columns)) == 3
    print("make_schedules tests passed!")

    rankings = data_cleaning.make_rankings()
    rankings_popped_columns = set(["scrape_date", "pos"])
    rankings_added_columns = set(["year", "month", "day"])
    assert rankings_popped_columns & set(rankings.columns) == set()
    assert len(rankings_added_columns & set(rankings.columns)) == 3
    print("make_rankings tests passed!")

    test_data = sf.add_home_and_away(
        data_cleaning.make_schedules(2020).slice(0, 1))
    test_data = data_cleaning.clean_for_training(test_data)
    assert (set(test_data.columns) & set("location")) == set()
    assert test_data.null_count().sum_horizontal().item() == 0
    print("clean_for_training tests passed!")

    print("data_cleaning tests passed!")


def visualize_overfitting() -> None:
    """
    Creates visualizations comparing an NflGamePredictor's
    profitability on 100 matches of training and testing data
    when the model is trained using 20, 40, 60, 80, and 100 epochs.
    The plots are saved to a file named model_fitting.png.
    """

    rosters = sf.get_all_schedules()
    cleaned_rosters = data_cleaning.clean_for_training(rosters)
    train_data = cleaned_rosters.slice(0, 1290)
    test_sample = cleaned_rosters.slice(1290, 100)
    test_features = torch.from_numpy(test_sample.select(model.TRAINING_FEATURES)
                                     .to_numpy().astype(np.float32, copy=False))
    train_sample = train_data.slice(1190, 100)
    train_features = torch.from_numpy(train_sample.select(model.TRAINING_FEATURES)
                                      .to_numpy().astype(np.float32, copy=False))
    betting_cols = ["total_line", "under_odds", "over_odds", "total",
                    "spread_line", "away_spread_odds", "home_spread_odds",
                    "result"]

    # Initializing DataFrames that will store the models' wealth
    matches = [i for i in range(len(test_sample))]
    train_wealth = pl.DataFrame(pl.Series(name="match", values=matches))
    test_wealth = pl.DataFrame(pl.Series(name="match", values=matches))
    device = model.DEVICE

    # Creating DataFrames that store the betting metrics for the
    # matches of interest
    test_betting_metrics = test_sample[betting_cols].clone()
    train_betting_metrics = train_sample[betting_cols].clone()

    # Simulating the betting strategies from model_profitability for
    # the five models described, and storing the models' wealth in
    # the training_wealth and test_wealth DataFrames
    for i in range(1, 6):
        predictor, _ = model.train_model(train_data, epochs=(i * 20),
                                         device=device)
        # Getting a model's predictions
        with torch.inference_mode():
            test_values = predictor(test_features).cpu().numpy()
            train_values = predictor(train_features).cpu().numpy()

        # Storing the model's wealth when it bets on test matches
        test_betting_metrics = test_betting_metrics.with_columns(
            pl.Series(name="predicted_diff", values=test_values[:, 2]),
            pl.Series(name="predicted_total", values=test_values[:, 3])
        )
        total_wealth = model_profitability.\
            get_total_profitability(test_betting_metrics)["wealth"]
        spread_wealth = model_profitability.\
            get_spread_profitability(test_betting_metrics)["wealth"]
        test_wealth = test_wealth.with_columns(
            pl.Series(total_wealth).alias(f"{i * 20}_epochs_total_wealth"),
            pl.Series(spread_wealth).alias(f"{i * 20}_epochs_spread_wealth")
        )

        # Storing the model's wealth when its bets on training matches
        train_betting_metrics = train_betting_metrics.with_columns(
            pl.Series(name="predicted_diff", values=train_values[:, 2]),
            pl.Series(name="predicted_total", values=train_values[:, 3])
        )
        total_wealth = model_profitability.\
            get_total_profitability(train_betting_metrics)["wealth"]
        spread_wealth = model_profitability.\
            get_spread_profitability(train_betting_metrics)["wealth"]
        train_wealth = train_wealth.with_columns(
            pl.Series(total_wealth).alias(f"{i * 20}_epochs_total_wealth"),
            pl.Series(spread_wealth).alias(f"{i * 20}_epochs_spread_wealth")
        )

    # Plotting the model's wealth after betting on training data
    # versus testing data
    fig, ax = plt.subplots(5, 2, figsize=(10, 50))
    for i in range(1, 6):
        j = i - 1

        # Plotting total wealth
        sns.lineplot(x="match",
                     y=f"{i * 20}_epochs_total_wealth",
                     data=train_wealth.to_pandas(), ax=ax[j][0],
                     label="train wealth")
        sns.lineplot(x="match",
                     y=f"{i * 20}_epochs_total_wealth",
                     data=test_wealth.to_pandas(), ax=ax[j][0],
                     label="test wealth")
        ax[j][0].set_xlabel("Match Number")
        ax[j][0].set_ylabel(f"{i * 20} Epochs Total Wealth")
        ax[j][0].set_title(f"Total Wealth Comparison for {i * 20} Epochs")
        ax[j][0].legend()

        # Plotting spread wealth
        sns.lineplot(x="match",
                     y=f"{i * 20}_epochs_spread_wealth",
                     data=train_wealth.to_pandas(), ax=ax[j][1],
                     label="train wealth")
        sns.lineplot(x="match",
                     y=f"{i * 20}_epochs_spread_wealth",
                     data=test_wealth.to_pandas(), ax=ax[j][1],
                     label="test wealth")
        ax[j][1].set_xlabel("Match Number")
        ax[j][1].set_ylabel(f"{i * 20} Epochs Spread Wealth")
        ax[j][1].set_title(f"Spread Wealth Comparison for {i * 20} Epochs")
        ax[j][1].legend()

    plt.savefig("model_fitting.png", bbox_inches="tight")


def main():
    test_schedule_formatting()
    test_data_cleaning()
    visualize_overfitting()


if __name__ == "__main__":
    main()
