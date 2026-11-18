"""
Cormac St John
CSE 163 A26
Implements functions and creates
visualizations showing which
factors of a model created with
model.py have the most impact on
its predictions
"""


import data_cleaning
import schedule_formatting
import model
import polars as pl
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np
import math


def create_importance_df(training_split: float,
                         num_epochs: int | None = None) -> pl.DataFrame:
    """
    Takes a float training_split that represetnts the percentage
    of NFL matches that will be used for training and an int or None
    num_epochs that represents the number of epochs on which a model
    will train, and trains an NflGamePredictor accordingly on NFL
    schedule data that has been cleaned for training, then tests which
    of the models features are most important in the model's predictions.
    Importance of a feature is measured by how much the mean squared error
    of a label's prediction changes as the feature's values are randomly modified.
    """
    rosters = schedule_formatting.get_all_schedules()
    cleaned_rosters = data_cleaning.clean_for_training(rosters)
    train_index = math.floor(len(cleaned_rosters.rows()) * training_split)
    train_data = cleaned_rosters.slice(0, train_index)
    test_data = cleaned_rosters.slice(train_index)
    device = model.DEVICE

    if num_epochs is None:
        trained_model, _ = model.train_model(train_data, device=device)
    else:
        trained_model, _ = model.train_model(train_data, 
                                             device=device,
                                             epochs=num_epochs)

    x = torch.as_tensor(
        test_data.select(model.TRAINING_FEATURES).to_numpy().astype(np.float32),
        device=device,
    )
    y = torch.as_tensor(
        test_data.select(model.TRAINING_LABELS).to_numpy().astype(np.float32),
        device=device,
    )

    importance = {}
    with torch.inference_mode():
        baseline = ((trained_model(x) - y) ** 2).mean(dim=0)

        for i, feature in enumerate(model.TRAINING_FEATURES):
            increases = []
            for _ in range(100):  # Repeat to reduce randomness
                shuffled = x.clone()
                shuffled[:, i] = x[torch.randperm(len(x), device=device), i]
                increases.append(abs(((trained_model(shuffled) - y) ** 2).mean(dim=0) - baseline))

            importance[feature] = torch.stack(increases).mean(dim=0).cpu().tolist()

    importance_df = pl.DataFrame({
        "feature": model.TRAINING_FEATURES,
        **{
            label: [importance[feature][i] for feature in model.TRAINING_FEATURES]
            for i, label in enumerate(model.TRAINING_LABELS)
        },
    })
    return importance_df


def create_importance_plot(data: pl.DataFrame, label: str, ax: plt.axes) -> None:
    """
    Takes a DataFrame of importance, a string label, and a plt.axes ax and
    creates a seaborn barplot if the importance of each feature in predicting
    label on the axis ax. The plot's x axis is labeled Features, its y axis is
    labeled Importance, and the plot is titled Importance of Features in
    Predicting LABEL, where LABEL is replaced by its respective input parameter
    """
    sns.barplot(x="feature", y=label, data=data.to_pandas(), ax=ax)
    ax.set_xlabel("Features")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylabel("Importance")
    ax.set_title(f"Importance of Features in Predicting {label.upper()}")


def main():
    """
    Creates a DataFrame using create_importance_df
    and creates seaborn plots of each training label
    using create_importance_plot. All of these plots
    are placed on a 2x2 figure that is saved to this
    file's directory as importance.png.
    """
    df = create_importance_df(training_split= 0.9, num_epochs=30)

    _, ax = plt.subplots(2, 2, figsize=(30, 25))
    for i, label in enumerate(model.TRAINING_LABELS):
        create_importance_plot(df,  label, ax[i // 2][i % 2])
    plt.savefig("importance.png", bbox_inches="tight")
    

if __name__ == "__main__":
    main()
