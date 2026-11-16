"""
Cormac St John
CSE 163 A26
Creates a machine learning model
that predicts betting metrics for
NFL matches using NFL match data
and fantasy rankings
"""


import polars as pl
import numpy as np
import torch
from torch import nn


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


class NFLGamePredictor(nn.Module):
    """Predict game scores and derived betting metrics from team features."""

    def __init__(self, feature_mean: torch.Tensor, feature_scale: torch.Tensor,
                 target_mean: torch.Tensor, target_scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_scale", feature_scale)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_scale", target_scale)
        self.network = nn.Sequential(
            nn.Linear(len(TRAINING_FEATURES), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(TRAINING_LABELS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = (features - self.feature_mean) / self.feature_scale
        normalized_predictions = self.network(normalized_features)
        return normalized_predictions * self.target_scale + self.target_mean


def train_model(data: pl.DataFrame, epochs: int = 100, batch_size: int = 32,
                learning_rate: float = 0.001,
                seed: int = 0,
                device: str | torch.device = "cuda"
                ) -> tuple[NFLGamePredictor, list[float]]:
    """Train on the requested device and return the model and epoch losses."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but CUDA is unavailable. "
            "Install a CUDA-enabled PyTorch build and check your GPU drivers.")

    required_columns = TRAINING_FEATURES + TRAINING_LABELS
    missing_columns = [column for column in required_columns
                       if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {missing_columns}")
    if data.is_empty():
        raise ValueError("Training data must contain at least one row")

    feature_values = data.select(TRAINING_FEATURES).to_numpy().astype(
        np.float32, copy=False)
    target_values = data.select(TRAINING_LABELS).to_numpy().astype(
        np.float32, copy=False)
    if not np.isfinite(feature_values).all() or not np.isfinite(target_values).all():
        raise ValueError("Training features and labels must all be finite")

    features = torch.from_numpy(feature_values).to(device)
    targets = torch.from_numpy(target_values).to(device)
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = targets.mean(dim=0)
    target_scale = targets.std(dim=0, unbiased=False).clamp_min(1e-6)

    torch.manual_seed(seed)
    predictor = NFLGamePredictor(feature_mean, feature_scale,
                                 target_mean, target_scale).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    normalized_targets = (targets - target_mean) / target_scale
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []

    predictor.train()
    for _ in range(epochs):
        indices = torch.randperm(len(features), generator=generator,
                                 device=device)
        epoch_loss = 0.0
        for start in range(0, len(features), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_features = features[batch_indices]
            batch_targets = normalized_targets[batch_indices]

            optimizer.zero_grad()
            predictions = predictor(batch_features)
            normalized_predictions = (
                predictions - target_mean) / target_scale
            loss = loss_function(normalized_predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_indices)

        losses.append(epoch_loss / len(features))

    predictor.eval()
    return predictor, losses
