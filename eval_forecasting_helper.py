"""This module evaluates the forecasted trajectories against the ground truth."""

import argparse
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pickle as pkl

from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from argoverse.map_representation.map_api import ArgoverseMap
from utils.baseline_config import FEATURE_FORMAT
from utils.baseline_utils import viz_predictions


def parse_arguments():
    """Parse command line arguments.

    Returns:
        parsed arguments
        
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics",
                        action="store_true",
                        help="If true, compute metrics")
    parser.add_argument("--gt", default="", type=str, help="path to gt file")
    parser.add_argument("--forecast",
                        default="",
                        type=str,
                        help="path to forecast file")
    parser.add_argument("--horizon",
                        default="",
                        type=int,
                        help="forecast horizon")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed Length")
    parser.add_argument("--miss_threshold",
                        default=2.0,
                        type=float,
                        help="Threshold for miss rate")
    parser.add_argument("--features",
                        default="",
                        type=str,
                        help="path to test features pkl file")
    parser.add_argument("--max_n_guesses",
                        default=0,
                        type=int,
                        help="Max number of guesses")
    parser.add_argument(
        "--prune_n_guesses",
        default=0,
        type=int,
        help="Pruned number of guesses of non-map baseline using map",
    )
    parser.add_argument(
        "--n_guesses_cl",
        default=0,
        type=int,
        help="Number of guesses along each centerline",
    )
    parser.add_argument("--n_cl",
                        default=0,
                        type=int,
                        help="Number of centerlines to consider")
    parser.add_argument("--viz",
                        action="store_true",
                        help="If true, visualize predictions")
    parser.add_argument(
        "--viz_seq_id",
        default="",
        type=str,
        help="Sequence ids for the trajectories to be visualized",
    )
    parser.add_argument(
        "--max_neighbors_cl",
        default=3,
        type=int,
        help="Number of neighbors obtained for each centerline by the baseline",
    )

    return parser.parse_args()


def get_city_names_from_features(features_df: pd.DataFrame) -> Dict[int, str]:
    """Get sequence id to city name mapping from the features.

    Args:
        features_df: DataFrame containing the features
    Returns:
        city_names: Dict mapping sequence id to city name

    """
    city_names = {}
    for index, row in features_df.iterrows():
        city_names[row["SEQUENCE"]] = row["FEATURES"][0][
            FEATURE_FORMAT["CITY_NAME"]]
    return city_names


def get_pruned_guesses(
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        city_names: Dict[int, str],
        gt_trajectories: Dict[int, np.ndarray],
) -> Dict[int, List[np.ndarray]]:
    """Prune the number of guesses using map.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        city_names: Dict mapping sequence id to city name.
        gt_trajectories: Ground Truth trajectories.

    Returns:
        Pruned number of forecasted trajectories.

    """
    args = parse_arguments()
    avm = ArgoverseMap()

    pruned_guesses = {}

    for seq_id, trajectories in forecasted_trajectories.items():

        city_name = city_names[seq_id]
        da_points = []
        for trajectory in trajectories:
            raster_layer = avm.get_raster_layer_points_boolean(
                trajectory, city_name, "driveable_area")
            da_points.append(np.sum(raster_layer))

        sorted_idx = np.argsort(da_points)[::-1]
        pruned_guesses[seq_id] = [
            trajectories[i] for i in sorted_idx[:args.prune_n_guesses]
        ]

    return pruned_guesses


def get_m_trajectories_along_n_cl(
        forecasted_trajectories: Dict[int, List[np.ndarray]]
) -> Dict[int, List[np.ndarray]]:
    """Given forecasted trajectories, get <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
    
    Returns:
        <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    """
    args = parse_arguments()
    selected_trajectories = {}
    for seq_id, trajectories in forecasted_trajectories.items():
        curr_selected_trajectories = []
        max_predictions_along_cl = min(len(forecasted_trajectories[seq_id]),
                                       args.n_cl * args.max_neighbors_cl)
        for i in range(0, max_predictions_along_cl, args.max_neighbors_cl):
            for j in range(i, i + args.n_guesses_cl):
                curr_selected_trajectories.append(
                    forecasted_trajectories[seq_id][j])
        selected_trajectories[seq_id] = curr_selected_trajectories
    return selected_trajectories


def viz_predictions_helper(
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        gt_trajectories: Dict[int, np.ndarray],
        features_df: pd.DataFrame,
        viz_seq_id: Union[None, List[int]],
) -> None:
    """Visualize predictions.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        gt_trajectories: Ground Truth trajectories.
        features_df: DataFrame containing the features
        viz_seq_id: Sequence ids to be visualized

    """
    args = parse_arguments()
    seq_ids = gt_trajectories.keys() if viz_seq_id is None else viz_seq_id
    for seq_id in seq_ids:
        gt_trajectory = gt_trajectories[seq_id]
        curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]
        input_trajectory = (
            curr_features_df["FEATURES"].values[0]
            [:args.obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
                "float"))
        output_trajectories = forecasted_trajectories[seq_id]
        candidate_centerlines = curr_features_df[
            "CANDIDATE_CENTERLINES"].values[0]
        city_name = curr_features_df["FEATURES"].values[0][
            0, FEATURE_FORMAT["CITY_NAME"]]

        gt_trajectory = np.expand_dims(gt_trajectory, 0)
        input_trajectory = np.expand_dims(input_trajectory, 0)
        output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
        candidate_centerlines = np.expand_dims(np.array(candidate_centerlines),
                                               0)
        city_name = np.array([city_name])
        viz_predictions(
            input_trajectory,
            output_trajectories,
            gt_trajectory,
            candidate_centerlines,
            city_name,
            show=True,
        )


if __name__ == "__main__":

    args = parse_arguments()

    with open(args.gt, "rb") as f:
        gt_trajectories: Dict[int, np.ndarray] = pkl.load(f)

    with open(args.forecast, "rb") as f:
        forecasted_trajectories: Dict[int, List[np.ndarray]] = pkl.load(f)

    with open(args.features, "rb") as f:
        features_df: pd.DataFrame = pkl.load(f)

    if args.metrics:

        city_names = get_city_names_from_features(features_df)

        # Get displacement error and dac on multiple guesses along each centerline
        if not args.prune_n_guesses and args.n_cl:
            forecasted_trajectories = get_m_trajectories_along_n_cl(
                forecasted_trajectories)
            num_trajectories = args.n_cl * args.n_guesses_cl

        # Get displacement error and dac on pruned guesses
        elif args.prune_n_guesses:
            forecasted_trajectories = get_pruned_guesses(
                forecasted_trajectories, city_names, gt_trajectories)
            num_trajectories = args.prune_n_guesses

        # Normal case
        else:
            num_trajectories = args.max_n_guesses

        compute_forecasting_metrics(
            forecasted_trajectories,
            gt_trajectories,
            city_names,
            num_trajectories,
            args.horizon,
            args.miss_threshold,
        )

    if args.viz:
        id_for_viz = None
        if args.viz_seq_id:
            with open(args.viz_seq_id, "rb") as f:
                id_for_viz = pkl.load(f)
        viz_predictions_helper(forecasted_trajectories, gt_trajectories,
                               features_df, id_for_viz)
