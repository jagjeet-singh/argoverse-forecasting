"""This module is used for Nearest Neighbor based baselines.

Example usage:
    $ python nn_train_test.py 
        --test_features ../data/forecasting_data_test.pkl 
        --train_features ../data/forecasting_data_train.pkl 
        --val_features ../data/forecasting_data_val.pkl ../../data/ 
        --use_map --use_delta --n_neigh 3 
        --traj_save_path forecasted_trajectories/nn_none.pkl
"""

import argparse
import numpy as np
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import time
import ipdb

import utils.baseline_utils as baseline_utils
from utils.nn_utils import Regressor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument("--test",
                        action="store_true",
                        help="Load the saved model and test")
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used.",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
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
        "--n_neigh",
        default=1,
        type=int,
        help=
        "Number of Nearest Neighbors to take. For map-based baselines, it is number of neighbors along each centerline.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help=
        "path to the pickle file where the model will be / has been saved.",
    )
    parser.add_argument(
        "--traj_save_path",
        required=True,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )

    return parser.parse_args()


def perform_k_nn_experiments(
        data_dict: Dict[str, Union[np.ndarray, pd.DataFrame, None]],
        baseline_key: str) -> None:
    """Perform various experiments using K Nearest Neighbor Regressor.

    Args:
        data_dict (dict): Dictionary of train/val/test data
        baseline_key: Key for obtaining features for the baseline

    """
    args = parse_arguments()

    # Get model object for the baseline
    model = Regressor()

    test_input = data_dict["test_input"]
    test_output = data_dict["test_output"]
    test_helpers = data_dict["test_helpers"]

    train_input = data_dict["train_input"]
    train_output = data_dict["train_output"]
    train_helpers = data_dict["train_helpers"]

    val_input = data_dict["val_input"]
    val_output = data_dict["val_output"]
    val_helpers = data_dict["val_helpers"]

    # Merge train and val splits and use K-fold cross validation instead
    train_val_input = np.concatenate((train_input, val_input))
    train_val_output = np.concatenate((train_output, val_output))
    train_val_helpers = np.concatenate([train_helpers, val_helpers])

    if args.use_map:
        print("####  Training Nearest Neighbor in NT frame  ###")
        model.train_and_infer_map(
            train_val_input,
            train_val_output,
            test_helpers,
            len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
            args,
        )

    else:
        print("####  Training Nearest Neighbor in absolute map frame  ###")
        model.train_and_infer_absolute(
            train_val_input,
            train_val_output,
            test_input,
            test_helpers,
            len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
            args,
        )


def main():
    """Load data and perform experiments."""
    args = parse_arguments()

    if not baseline_utils.validate_args(args):
        return

    np.random.seed(100)

    # Get features
    if args.use_map and args.use_social:
        baseline_key = "map_social"
    elif args.use_map:
        baseline_key = "map"
    elif args.use_social:
        baseline_key = "social"
    else:
        baseline_key = "none"

    # Get data
    data_dict = baseline_utils.get_data(args, baseline_key)

    # Perform experiments
    start = time.time()
    perform_k_nn_experiments(data_dict, baseline_key)
    end = time.time()
    print(f"Completed experiment in {(end-start)/60.0} mins")


if __name__ == "__main__":
    main()
