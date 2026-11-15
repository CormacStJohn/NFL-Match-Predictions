"""
Cormac St John
CSE 163 A26
Creates a machine learning model
that predicts betting metrics for
NFL matches using NFL match data
and fantasy rankings
"""


import nflreadpy as nfl
import polars as pl
import numpy as np
import torch
import data_cleaning
import schedule_formatting
import sklearn as skl


TRAINING_SEASONS = [x for x in range(2020, 2027)]
TRAINING_FEATURES = ['week', 'location', 'away_rest', 'home_rest',
                     'home_WR_0', 'home_WR_1', 'home_DB_0', 'home_DB_1',
                     'home_DB_2', 'home_DB_3', 'home_DL_0', 'home_DL_1',
                     'home_DL_2', 'home_DL_3', 'home_TE_0', 'home_K_0',
                     'home_LB_0', 'home_LB_1', 'home_LB_2', 'home_RB_0',
                     'home_RB_1', 'home_QB_0', 'away_WR_0', 'away_WR_1',
                     'away_DB_0', 'away_DB_1', 'away_DB_2', 'away_DB_3',
                     'away_DL_0', 'away_DL_1', 'away_DL_2', 'away_DL_3',
                     'away_TE_0', 'away_K_0', 'away_LB_0', 'away_LB_1',
                     'away_LB_2', 'away_RB_0', 'away_RB_1', 'away_QB_0']
TRAINING_LABELS = ['away_score', 'home_score', 'result', 'total']


# A list of abbreviations for positions on a football team.
POSITIONS: list[str] = ["WR", "DB", "DL", "TE", "K", "LB", "RB", "QB"]


# A dictionary consisting of abbreviations for positions, as 
# well as the number of players on the field that have that
# position at a given time.
POSITION_AMOUNTS: dict[str, int] = {"WR": 2, "OL": 5, "DB": 4, "DL": 4,
                                    "TE": 1, "K": 1, "LB": 3, "LS": 1,
                                    "RB": 2, "P": 1, "QB": 1}


# A DataFrame of match schedules for the seasons of interest
SCHEDULES = data_cleaning.make_schedules(TRAINING_SEASONS)


rosters = schedule_formatting.add_home_and_away(SCHEDULES)
cleaned_rosters = data_cleaning.clean_for_training(rosters)

