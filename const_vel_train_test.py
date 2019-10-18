"""const_vel_train_test.py runs a constant velocity baseline.

Example usage:
    python const_vel_test.py --test_features data/features/forecasted_features_test.pkl 
        --obs_len 20 --pred_len 30 --traj_save_path forecasted_trajectories/const_vel.pkl
"""
import argparse
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import mean_squared_error

from utils.baseline_config import FEATURE_FORMAT
import utils.baseline_utils as baseline_utils


def parse_arguments():
    """Parse Arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--traj_save_path",
        required=True,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    return parser.parse_args()


def get_mean_velocity(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get mean velocity of the observed trajectory.

    Args:
        coords: Coordinates for the trajectory
    Returns:
        Mean velocity along x and y

    """
    vx, vy = (
        np.zeros((coords.shape[0], coords.shape[1] - 1)),
        np.zeros((coords.shape[0], coords.shape[1] - 1)),
    )

    for i in range(1, coords.shape[1]):
        vx[:, i - 1] = (coords[:, i, 0] - coords[:, i - 1, 0]) / 0.1
        vy[:, i - 1] = (coords[:, i, 1] - coords[:, i - 1, 1]) / 0.1
    vx = np.mean(vx, axis=1)
    vy = np.mean(vy, axis=1)

    return vx, vy


def predict(obs_trajectory: np.ndarray, vx: np.ndarray, vy: np.ndarray,
            args: Any) -> np.ndarray:
    """Predict future trajectory given mean velocity.

    Args:
        obs_trajectory: Observed Trajectory
        vx: Mean velocity along x
        vy: Mean velocity along y
        args: Arguments to the baseline
    Returns:
        pred_trajectory: Future trajectory

    """
    pred_trajectory = np.zeros((obs_trajectory.shape[0], args.pred_len, 2))

    prev_coords = obs_trajectory[:, -1, :]
    for i in range(args.pred_len):
        pred_trajectory[:, i, 0] = prev_coords[:, 0] + vx * 0.1
        pred_trajectory[:, i, 1] = prev_coords[:, 1] + vy * 0.1
        prev_coords = pred_trajectory[:, i]

    return pred_trajectory


def forecast_and_save_trajectory(obs_trajectory: np.ndarray,
                                 seq_id: np.ndarray, args: Any) -> None:
    """Forecast future trajectory and save it.

    Args:
        obs_trajectory: Observed trajectory
        seq_id: Sequence ids
        args: Arguments to the baseline

    """
    vx, vy = get_mean_velocity(obs_trajectory)
    pred_trajectory = predict(obs_trajectory, vx, vy, args)

    forecasted_trajectories = {}

    for i in range(pred_trajectory.shape[0]):
        forecasted_trajectories[seq_id[i]] = [pred_trajectory[i]]

    with open(args.traj_save_path, "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def main():
    """Main."""
    args = parse_arguments()

    df = pd.read_pickle(args.test_features)

    feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
    seq_id = df["SEQUENCE"].values

    obs_trajectory = np.stack(
        df["FEATURES"].values)[:, :args.obs_len, feature_idx].astype("float")

    forecast_and_save_trajectory(obs_trajectory, seq_id, args)


if __name__ == "__main__":
    main()
